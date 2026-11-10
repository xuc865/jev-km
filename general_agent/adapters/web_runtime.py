"""Shared Runtime adapter: public state -> legal input candidates -> browser input."""
from __future__ import annotations
import base64
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import time

from webenv.host import BrowserHost
from webenv.timing import timing_contract
from webenv.navigation import reachable_targets, describe_reachability
from webenv.dom import terminal as dom_terminal, phase as dom_phase, annotate_candidates
from ..types import Action, Observation
from ..learning import render_lessons
from ..scene_memory import ObservedScenes
from ..progress import ObservedProgress
from .. import progress_probe

ROOT = Path(__file__).resolve().parents[2]


def field(value, path):
    for name in path.split('.'):
        if not isinstance(value, dict): return None
        value = value.get(name)
    return value


def matches(state, conditions):
    return all(field(state, key) in expected if isinstance(expected, list)
               else field(state, key) == expected for key, expected in conditions.items())


def signature(state):
    return hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


def frame_signature(png: bytes) -> str:
    """Coarse hash of the rendered pixels.

    signature() hashes the DOM snapshot, and a canvas game exposes no DOM nodes
    at all, so its hash stayed identical for an entire episode. The recorded
    `before`/`after` were then equal on every step and last_execution reported
    changed=false forever, which tells the model its actions have no effect
    when in fact the screen was moving. Pixel content has to be part of the key
    for those games.
    """
    if not png:
        return ""
    try:
        from io import BytesIO
        from PIL import Image
        image = Image.open(BytesIO(png)).convert("L").resize((32, 20))
        return hashlib.sha256(bytes(image.getdata())).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(png).hexdigest()[:16]



class WebRuntimeAdapter:
    name = 'HTML5 Runtime'
    def __init__(self, game_id, goal='', entry=None, config=None):
        self.game_id = game_id
        self.pack_dir = ROOT / 'knowledge' / game_id
        self.config = config if config is not None else json.loads((self.pack_dir / 'runtime.json').read_text(encoding='utf-8'))
        self.entry = entry or {}
        self.goal = goal or self.config.get('goal', '')
        self.host = None
        self.artifacts = None
        self._state = {}
        self._page = 0
        self.preview = ''
        self.last_execution = {}
        self.executions = []
        self.steps = 0
        self._terminal_settled = False
        self.scene_memory=ObservedScenes(self.config.get('memory'))
        self.progress=ObservedProgress(self.config.get('progress_fields'))
        # `self.progress` is already the pack-declared outcome tracker, so the
        # catalogue metric keeps its own names. Keeping them separate means a
        # pack with both still reports the hand-written fields for the model and
        # the catalogue metric for training-data filtering.
        self.metric: dict = progress_probe.pending(game_id, None)
        self.metric_best = None
        self.metric_name = ''
        self._metric_injected = False

    def reset(self):
        self._terminal_settled = False
        self.scene_memory=ObservedScenes(self.config.get('memory'))
        self.progress=ObservedProgress(self.config.get('progress_fields'))
        # `self.progress` is already the pack-declared outcome tracker, so the
        # catalogue metric keeps its own names. Keeping them separate means a
        # pack with both still reports the hand-written fields for the model and
        # the catalogue metric for training-data filtering.
        # This is not the constructor: `game_id` is not in scope here, only
        # `self.game_id` is. Using the bare name raised NameError inside reset(),
        # so every pack-backed game died before its first observation and
        # recorded no progress at all.
        self.metric: dict = progress_probe.pending(self.game_id, None)
        self.metric_best = None
        self.metric_name = ''
        self._metric_injected = False
        folder = Path(self.artifacts) if self.artifacts else Path(tempfile.mkdtemp(prefix='jevgamer-runtime-'))
        self.artifacts = folder
        self.host = BrowserHost(self.entry['entry_path'], self.config, self.pack_dir, folder,
        seed=int(os.environ.get('JEVGAMER_SEED') or self.config.get('seed', 42)), record=self.config.get('record_video', True))
        self.host.start()
        self.metric = None
        self.metric_best = None
        self.metric_name = ''
        self._metric_injected = progress_probe.inject(self.host, self.game_id)

    def _metric_record(self) -> dict:
        """Sample the catalogue progress metric and keep the episode peak.

        A pack that declares `progress_fields` gets its own tracking on top of
        this; the two answer different questions, so neither overrides the other.
        """
        record = progress_probe.read(self.host) if self.host is not None else None
        if record is None:
            self.metric = progress_probe.pending(self.game_id, self.metric_best)
            return dict(self.metric)
        best = record.get('best')
        if isinstance(best, (int, float)):
            lower_is_better = str(record.get('sense') or 'high') == 'low'
            if self.metric_best is None:
                self.metric_best = best
            elif lower_is_better:
                self.metric_best = min(self.metric_best, best)
            else:
                self.metric_best = max(self.metric_best, best)
        self.metric_name = str(record.get('metric') or self.metric_name)
        record['source'] = 'game'
        record['episode_best'] = self.metric_best
        self.metric = dict(record)
        return dict(self.metric)

    def _scene_signature(self, state) -> str:
        """DOM digest plus rendered-pixel digest, for change detection only.

        This must NOT be used to validate that an observation is current: pixels
        change every frame in an animating game, so a signature containing them
        is stale by the time the action executes and every action gets rejected.
        It exists for the view/feedback record, where sensitivity is the point.
        """
        try:
            shot = self.host.screenshot()
        except Exception:
            shot = b""
        return signature(state) + ":" + frame_signature(shot)

    def _read(self):
        state = self.host.snapshot()
        if self.config.get('observation') == 'dom':
            done,success=dom_terminal(state.get('headings',[]),self.config)
            state={'phase':dom_phase(state,self.config),'done':done,'success':success,
                   'screen':state.get('text',''),'headings':state.get('headings',[]),'public_fields':state.get('fields',{}),'_dom':state,
                   'busy':bool(self.config.get('busy_pattern') and re.search(self.config['busy_pattern'],state.get('text',''))),
                   'flags':{key:bool(re.search(pattern,state.get('text',''))) for key,pattern in self.config.get('text_flags',{}).items()},
                   'integrity':{'built_in_ai':False}}
        if not isinstance(state, dict): raise RuntimeError('Runtime snapshot must be an object')
        if state.get('navigation') and self.config.get('navigation'):
            state={**state,'navigation':describe_reachability(state['navigation'],self.config['navigation']['keys'])}
        for key, expected in self.config.get('integrity', {}).items():
            actual = field(state, 'integrity.' + key)
            if actual != expected:
                raise RuntimeError(f'Runtime integrity failed: {key}={actual!r}, expected {expected!r}')
        return state

    def _actions(self, state):
        if state.get('navigation') and self.config.get('navigation'):
            state={**state,'navigation':describe_reachability(state['navigation'],self.config['navigation']['keys'])}
        actions = []
        if self.config.get('observation') == 'dom':
            current=signature(state)
            for item in annotate_candidates(state.get('_dom',{}).get('candidates',[]),state.get('flags',{}),self.config.get('candidate_annotations',[])):
                actions.append(Action(item['id'],item['label'],kind='web_scroll' if item.get('kind')=='scroll' else 'dom_click',
                    seconds=0,params={'selector':item['selector'],'direction':item.get('direction'),'semantic_role':item.get('role'),'signature':current}))
            if not state.get('done'):
                actions.append(Action('wait','等待页面动画或敌方行动结束，再重新观察',kind='wait',seconds=0.6,
                                      params={'frames':36,'signature':current}))
            return actions
        for item in self.config.get('controls', []):
            if not matches(state, item.get('when', {})): continue
            if item.get('selector'):
                locator = self.host.frame.locator(item['selector'])
                if not locator.is_visible() or not locator.is_enabled(): continue
            actions.append(Action(item['id'], item['label'], kind=item.get('kind', 'web_keys'),
                                  keys=tuple(item.get('keys', [])),
                                  seconds=item.get('frames',1)*self.config.get('time',{}).get('dt_ms',1000/60)/1000,
                                  params={'selector': item.get('selector'), 'frames': item.get('frames', 1)}))
        for item in self.config.get('dom_controls', []):
            if not matches(state, item.get('when', {})): continue
            locators = self.host.frame.locator(item['selector']).all()
            for index, locator in enumerate(locators):
                if not locator.is_visible() or not locator.is_enabled(): continue
                label = locator.inner_text().strip()[:700]
                actions.append(Action(item.get('id_prefix','choice')+'_'+str(index), label,
                    kind='dom_click', seconds=0, params={'selector':item['selector'],'nth':index,'frames':1}))
        nav = state.get('navigation')
        if nav:
            settings = self.config['navigation']
            targets = reachable_targets(nav, settings['keys'])
            capacity = max(1, 31 - len(actions))
            pages = max(1, (len(targets) + capacity - 1) // capacity)
            self._page %= pages
            selected = targets[self._page * capacity:(self._page + 1) * capacity]
            for target in selected:
                actions.append(Action(target['id'], f"({target['x']},{target['y']}) {target['label']}；空地路径{target['steps']}步",
                    kind='web_path',
                    seconds=target['steps']*settings.get('frames_per_key',2)*self.config.get('time',{}).get('dt_ms',1000/60)/1000, params={'keys': target['keys'], 'positions': target['positions'],
                    'frames': settings.get('frames_per_key', 2)}))
            if pages > 1:
                actions.append(Action('next_targets', f'查看下一页可达目标（当前{self._page+1}/{pages}页）',
                                      kind='interface', seconds=0))
        current = signature(state)
        for action in actions: action.params['signature'] = current
        if len(actions) > 32: raise RuntimeError('Knowledge pack exposes more than 32 controls')
        return actions

    def observe(self):
        state = self._read()
        quiet=float(self.config.get('observation_quiet_seconds',0))
        changed=time.monotonic();previous=self._scene_signature(state);deadline=changed+20
        while quiet and not state.get('done') and time.monotonic()-changed<quiet:
            if time.monotonic()>deadline:raise RuntimeError('DOM observation did not settle')
            self.host.page.wait_for_timeout(100)
            state=self._read();current=self._scene_signature(state)
            if current!=previous:changed=time.monotonic();previous=current
        advanced = 0
        while state.get('busy') and not state.get('done'):
            if advanced >= 1800: raise RuntimeError('Non-interactive animation exceeded 30 virtual seconds')
            self.host.advance(1)
            advanced += 1
            state = self._read()
        terminal_frames = 0
        if state.get('done') and not self._terminal_settled:
            terminal_frames=max(0,int(self.config.get('time',{}).get('terminal_frames',0)))
            if terminal_frames:
                self.host.advance(terminal_frames)
                state=self._read()
            self._terminal_settled=True
        observed_at=time.monotonic()
        self._state = state
        self.preview = 'data:image/png;base64,' + base64.b64encode(self.host.screenshot()).decode()
        if state.get('done'):
            (self.artifacts / 'final.png').write_bytes(base64.b64decode(self.preview.split(',',1)[1]))
        prior_scenes=self.scene_memory.remember(state,self.steps)
        review = self.config.get('review_fields', ['phase'])
        headings = [json.dumps({key:field(state,key) for key in review},ensure_ascii=False,sort_keys=True)]
        lessons=render_lessons(self.pack_dir,str(state.get('phase') or ''))
        timing=timing_contract(self.config)
        controls=self.config.get('instructions','')+'\nRuntime timing contract: '+timing['label']
        # The text state of a canvas game is empty, so without an explicit view
        # record every observation looks identical and the model cannot tell a
        # dead action from a slow one.
        view = {
            "signature": frame_signature(base64.b64decode(self.preview.split(',', 1)[1])),
            "changed_since_last_step": None,
            "distinct_views": getattr(self, "_distinct_views", 0),
            "frames_observed": getattr(self, "_frames_observed", 0),
        }
        last_sig = getattr(self, "_last_view_sig", "")
        if view["signature"]:
            view["changed_since_last_step"] = bool(last_sig) and view["signature"] != last_sig
            if view["signature"] != last_sig:
                self._distinct_views = getattr(self, "_distinct_views", 0) + 1
            self._last_view_sig = view["signature"]
            self._frames_observed = getattr(self, "_frames_observed", 0) + 1
            view["distinct_views"] = self._distinct_views
            view["frames_observed"] = self._frames_observed
        full = {'game': self.game_id, 'phase': state.get('phase'), 'headings': headings,
                'view': view,
                'timing_contract':timing,'instructions': controls+('\nEvidence-backed experience hypotheses:\n'+lessons if lessons else ''), 'runtime': state,
                'metric_progress': self._metric_record(),
                'last_execution': self.last_execution}

        if prior_scenes:full['observed_scenes']=prior_scenes
        progress=self.progress.observe(state,self.steps)
        if progress:full['progress_evidence']=progress
        fast_fields = self.config.get('fast_fields')
        if fast_fields:
            full['fast_state'] = {key:field(state,key) for key in fast_fields}
            full['fast_state']['control_instructions']=controls
            # The prompt is built from fast_state, so an execution record kept
            # only on `full` is invisible to the model: it sees the world change
            # but never learns which action caused the change. Carry the feedback
            # and the view delta into the section that is actually read.
            full['fast_state']['last_execution'] = self.last_execution
            full['fast_state']['view'] = view
            if progress:full['fast_state']['progress_evidence']=progress
            if lessons:full['fast_state']['experience_hypotheses']=lessons[:1000]
        actions=[] if state.get('done') else self._actions(state)
        for action in actions:action.params['observed_at_monotonic']=observed_at
        return Observation(actions=actions, image=self.preview, state=full,
            level=int(state.get('floor') or state.get('level') or 1),
            done=state.get('done') is True, success=state.get('success') is True,
            raw={'signature':signature(state), 'animation_frames':advanced, 'terminal_frames':terminal_frames,'observed_at_monotonic':observed_at, 'artifacts':str(self.artifacts)})

    def show_decision(self, decision, strategy=None):
        strategy = strategy or {}
        ranking = sorted(decision.probabilities.items(), key=lambda item:item[1], reverse=True)
        labels = {action.id:action.label for action in self._actions(self._state)}
        self.host.display({'run':self.game_id + ' · seed ' + str(self.config.get('seed',42)),
            'status':f'第 {self.steps+1} 步 · {decision.model} · {decision.by}\n'+strategy.get('outer_model','')+'\n'+labels.get(decision.action_id,decision.action_id)+'\n完整性检查：'+json.dumps(self._state.get('integrity',{}),ensure_ascii=False),
            'strategy':strategy.get('stage_goal','')+'\n'+strategy.get('play_instruction',''),
            'scores':'\n'.join(f'{prob:.1%} {labels.get(key,key)}' for key,prob in ranking[:10]),
            'execution':json.dumps(self.last_execution,ensure_ascii=False,indent=2)})

    def act(self, action):
        before = self._read()
        before_sig = signature(before)
        before_pixel = frame_signature(self.host.screenshot() if hasattr(self.host, 'screenshot') else b'')
        captured=action.params.get('observed_at_monotonic')
        observation_age_ms=round((time.monotonic()-captured)*1000) if isinstance(captured,(int,float)) else None
        if action.params.get('signature') != before_sig:
            self.last_execution = {'action':action.id,'ok':False,'error':'stale observation; reobserve',
                'observation_age_ms':observation_age_ms,'expected_signature':action.params.get('signature'),'current_signature':before_sig}
            self.executions.append(self.last_execution.copy())
            return
        keys_executed = []
        if action.kind == 'interface': self._page += 1
        elif action.kind == 'dom_click': self.host.click(action.params['selector'], action.params.get('nth')); self.host.advance(int(self.config.get('time',{}).get('action_frames',1)))
        elif action.kind == 'web_scroll': self.host.scroll(action.params['selector'],action.params.get('direction','down'))
        elif action.kind == 'wait': self.host.advance(action.params.get('frames',36))
        elif action.kind == 'web_path':
            for index, key in enumerate(action.params['keys']):
                self.host.keys([key], action.params['frames']); keys_executed.append(key)
                after = self._read()
                if after.get('phase') != before.get('phase') or after.get('done'): break
                nav = after.get('navigation') or {}
                if index < len(action.params['keys'])-1 and nav.get('position') != action.params['positions'][index]: break
        elif action.kind == 'web_keys':
            self.host.keys(action.keys, action.params['frames']); keys_executed = list(action.keys)
        else: raise ValueError('Unknown runtime action kind: ' + action.kind)
        settle=float(self.config.get('action_settle_seconds',0))
        if settle:self.host.page.wait_for_timeout(settle*1000)
        after = self._read()
        after_sig = signature(after)
        after_pixel = frame_signature(self.host.screenshot() if hasattr(self.host, 'screenshot') else b'')
        self.steps += 1
        self.last_execution = {'action':action.id,'label':action.label,'ok':True,
            'before':before_pixel,'after':after_pixel,'changed':bool(before_pixel) and before_pixel!=after_pixel,
            'dom_changed':before_sig!=after_sig,
            'keys_executed':keys_executed,'phase_after':after.get('phase'),'observation_age_ms':observation_age_ms}
        self.executions.append(self.last_execution.copy())
        # Sample right after the step. The game clock keeps running between
        # observations, so a peak reached here would otherwise be recorded low.
        self._metric_record()
        self.last_execution['metric_after'] = dict(self.metric or {})


    def close(self):
        if self.host:
            if self._state.get('done'):self.host.page.wait_for_timeout(2000)
            self.host.close()
            self.host = None
