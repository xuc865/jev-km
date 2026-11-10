"""Slow-model postmortems persisted through the existing Memory store.

Lessons are evidence-backed prompt data. They never execute game actions,
change game rules, overwrite goals, or act as a deterministic fallback.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from agent_harness import Thread
from experience import Memory, priority_rank

ROOT=Path(__file__).resolve().parents[1]
COMMON=ROOT/'knowledge'/'common'
REVIEW_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

SYSTEM='''You review an autonomous JEV + DeepSeek game episode. Ground every claim in supplied trace steps or the authoritative terminal result. Do not invent success, routes, mechanics or stable win rates from one run. Classify lessons as common only when they concern transferable observation, execution, progress validation or planning discipline; game combat/build/routing tactics belong to game scope. Lessons are conditional hypotheses used as model context, never executable policies. Preserve the locked player identity and goal. Return JSON only:
{"summary":"...","facts":[{"id":"short-stable-id","scope":"common|game","phase":"any|observed phase","instruction":"conditional advice under 500 chars","why":"evidence-based explanation","evidence_steps":[1,2],"confidence":0.0}],"limitations":["..."]}
Keep summary under 180 characters. Return at most 4 facts total: at most 2 common and 2 game. Keep each instruction under 220 characters and each why under 120 characters. Every fact must cite at least one supplied evidence step. Visible navigation targets are not necessarily reachable legal actions; inspect legal_actions and executable_now when present. Do not claim a visible resource is freely reachable without that evidence. Only propose declared controls; if a necessary control is missing, describe the observation/execution gap instead of inventing a back or action API. Never recommend built-in autoplay, hidden state, score shortcuts, modified game values or assistant intervention.'''


def render_lessons(pack_dir: Path, phase: str) -> str:
    facts=Memory(pack_dir).facts + Memory(COMMON,'web-runtime.experience.json').facts
    selected=[fact for fact in facts if fact.get('kind')=='lesson'
              and (fact.get('when') or fact.get('if') or {}).get('phase','any') in {'any',phase}]
    selected.sort(key=lambda fact:(priority_rank(fact.get('priority')),-int(fact.get('support') or 0)))
    lines=[]
    for fact in selected[:8]:
        text=str((fact.get('do') or fact.get('then') or {}).get('instruction') or '')
        if text:lines.append(f"[{fact.get('id')}; support={fact.get('support',1)}] {text}")
    return '\n'.join(lines)[:3000]


def evidence_from_trace(rows, executions=None):
    if not rows:return []
    def spread(items,count):
        if count<=0:return []
        if len(items)<=count:return list(items)
        if count==1:return [items[len(items)//2]]
        return [items[round(index*(len(items)-1)/(count-1))] for index in range(count)]
    # Keep outcomes, causal transition windows, and within-phase trajectories.
    # Sampling only headings disproportionately shows resets/serves while
    # hiding the actual play that caused the next score or phase change.
    selected={index for index in [0,1,*range(len(rows)-4,len(rows))] if 0<=index<len(rows)}
    events=[];failures=[];last_phase=None
    for index,row in enumerate(rows):
        state=row['observation'].get('state') or {}
        phase=(state.get('phase'),tuple(state.get('headings') or []))
        if last_phase is not None and phase!=last_phase:events.append(index)
        last_phase=phase
        if (row.get('last_execution') or {}).get('ok') is False:failures.append(index)
    for index in spread(failures,4):
        selected.update(point for point in [index-1,index] if point>=0)
    for index in spread(events,min(6,(32-len(selected))//3)):
        selected.update(range(max(0,index-2),index+1))
    remaining=[index for index in range(len(rows)) if index not in selected]
    selected.update(spread(remaining,32-len(selected)))
    indices=sorted(selected)
    evidence=[]
    for index in indices:
        row=rows[index];obs=row['observation'];state=obs.get('state') or {};runtime=state.get('runtime') or {}
        screen=str(state.get('screen') or runtime.get('screen') or runtime.get('visible_text') or '')
        if len(screen)>2400:screen=screen[:1200]+'\n[...middle omitted...]\n'+screen[-1200:]
        decision=row['decision'];action=next((item for item in obs.get('actions',[]) if item['id']==decision['action_id']),{})
        # Trace feedback acknowledges the preceding row. Align the chosen action
        # with its own execution rather than asking a reviewer to infer the shift.
        execution={}
        step=row['step']
        if executions is not None and isinstance(step,int) and 0<=step<len(executions):
            candidate=executions[step]
            if candidate.get('action')==decision.get('action_id'):execution=candidate
        elif index+1<len(rows):
            candidate=rows[index+1].get('last_execution') or {}
            if candidate.get('action')==decision.get('action_id'):execution=candidate
        evidence.append({'step':row['step'],'phase':state.get('phase'),'headings':state.get('headings'),
            'progress_evidence':state.get('progress_evidence',{}),'visible_screen':screen,'runtime':{key:value for key,value in runtime.items()
                if key not in {'screen','visible_text','_dom'}},
            'action':action.get('label',decision['action_id']),'by':decision['by'],
            'legal_actions':[{'id':item['id'],'label':item.get('label','')[:120],
                              'kind':item.get('kind'),'seconds':item.get('seconds')}
                             for item in obs.get('actions',[])],
            'execution':{key:value for key,value in execution.items() if key!='after_text'},
            'execution_alignment':'chosen action at this step' if execution else 'no matching execution recorded'})
    return evidence


def merge_reflection(game: str, run_id: str, parsed: dict, evidence: list, folder: Path, scopes=None):
    allowed={item['step'] for item in evidence};accepted=[]
    for raw in parsed.get('facts',[])[:12]:
        refs=raw.get('evidence_steps') or []
        if not refs or any(type(step) is not int or step not in allowed for step in refs):continue
        scope=raw.get('scope')
        if scope not in {'common','game'} or (scopes is not None and scope not in scopes):continue
        instruction=str(raw.get('instruction') or '').strip()[:600]
        if not instruction:continue
        phase=str(raw.get('phase') or 'any')[:80]
        identity=str(raw.get('id') or hashlib.sha256(instruction.encode()).hexdigest()[:12])[:80]
        topic=('web-runtime-' if scope=='common' else game+'-')+identity
        record={'id':topic,'kind':'lesson','when':{'phase':phase,'topic':topic},
                'do':{'instruction':instruction},'why':str(raw.get('why') or '')[:800],
                'source':'deepseek-reflection','priority':'normal'}
        memory=Memory(COMMON,'web-runtime.experience.json') if scope=='common' else Memory(ROOT/'knowledge'/game)
        existing=next((item for item in memory.facts if (item.get('when') or item.get('if'))==record['when']),None)
        if existing and any(proof.get('run_id')==run_id for proof in existing.get('evidence',[])):
            accepted.append(existing);continue
        memory.merge([record])
        fact=next(item for item in memory.facts if (item.get('when') or item.get('if'))==record['when'])
        trace_path=folder/'trace.jsonl'
        try:trace_reference=str(trace_path.relative_to(ROOT))
        except ValueError:trace_reference=str(trace_path)
        proof={'run_id':run_id,'steps':refs,'trace':trace_reference,
               'observations':[item for item in evidence if item['step'] in refs]}
        if proof not in fact.get('evidence',[]):fact.setdefault('evidence',[]).append(proof)
        fact.update(scope=scope,family='web-runtime',always=scope=='common',
                    confidence=max(0,min(1,float(raw.get('confidence') or 0))) if isinstance(raw.get('confidence'),(int,float)) else 0)
        memory.save();accepted.append(fact)
    return accepted


def distill(folder: Path, chat_fn=None, persist=True, persist_scopes=None):
    import harness
    from .outer import OuterEscalation
    folder=folder.resolve();summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (folder/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
    if not rows:raise ValueError('No model decisions to review')
    evidence=evidence_from_trace(rows,summary.get('executions'))
    result={key:value for key,value in (summary.get('result') or {}).items() if key!='decisions'}
    terminal=summary.get('final_visible_frame') or {}
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8')) if (folder/'protocol.json').is_file() else {}
    payload={'protocol':protocol,'game':summary['game'],'locked_goal':summary['goal'],'result':result,'error':summary.get('error'),
             'terminal':terminal,'decision_counts':dict(Counter(row['decision']['by'] for row in rows)),
             'evidence':evidence,'note':'Each evidence execution is aligned with its chosen action at that same step. Missing matching execution is explicitly marked; do not attribute preceding feedback to this action.'}
    thread=Thread(history_turns=4)
    for item in evidence[-4:]:
        turn=thread.begin_turn(item['step']);turn.add('observation',json.dumps(item,ensure_ascii=False))
        turn.add('agent_message','Historical executed action: '+item['action']);thread.complete_turn()
    messages=thread.messages_for_prompt(SYSTEM+'\nLocked root goal: '+summary['goal'],
                                        json.dumps(payload,ensure_ascii=False))
    runtime_config=protocol.get('runtime_config') or {}
    generation=runtime_config.get('review_generation',runtime_config.get('slow_generation',{}))
    response=(chat_fn or harness.chat)(messages,harness.TOKEN,model=harness.SUPERVISOR,**generation)
    if response.get('_error'):raise RuntimeError(response['_error'])
    text=str(((response.get('choices') or [{}])[0].get('message') or {}).get('content') or '')
    parsed=OuterEscalation._parse(text)
    if not parsed:raise ValueError('Reflection returned invalid JSON')
    report={'game':summary['game'],'run_id':summary['run_id'],'at':datetime.now(timezone.utc).isoformat(),
            'author_model':response.get('model') or harness.SUPERVISOR,'usage':response.get('usage'),
            'generation':harness.chat_generation(**generation),
            'review_source_sha256':REVIEW_SOURCE_SHA256,
            'authoritative_result':result,'reflection':parsed,'evidence':evidence}
    (folder/'reflection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if persist:
        report['accepted_facts']=merge_reflection(summary['game'],summary['run_id'],parsed,evidence,folder,persist_scopes)
        report['persistence_scope']=sorted(persist_scopes) if persist_scopes is not None else ['common','game']
        report['deferred_facts']=[fact for fact in parsed.get('facts',[])
                                  if persist_scopes is not None and fact.get('scope') not in persist_scopes]
        (folder/'reflection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_folder',type=Path)
    parser.add_argument('--no-persist',action='store_true')
    parser.add_argument('--game-only',action='store_true',help='Defer common-memory updates while other titles are running')
    args=parser.parse_args()
    report=distill(args.run_folder,persist=not args.no_persist,persist_scopes={'game'} if args.game_only else None)
    print(json.dumps({'game':report['game'],'model':report['author_model'],
                      'accepted_facts':len(report.get('accepted_facts',[])),
                      'reflection':str(args.run_folder/'reflection.json')},ensure_ascii=False))

if __name__=='__main__':main()
