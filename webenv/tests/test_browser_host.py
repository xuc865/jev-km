"""Opt-in browser contract test on a synthetic fixture, never a game clear."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

from webenv.host import BrowserHost

@unittest.skipUnless(os.environ.get('JEVGAMER_BROWSER_TESTS') == '1' and importlib.util.find_spec('playwright'),
                     'opt-in browser runtime test')
class BrowserHostTests(unittest.TestCase):
    def test_trusted_keyboard_frozen_clock_and_real_video(self):
        with tempfile.TemporaryDirectory(prefix='jev-host-contract-') as folder:
            root=Path(folder)
            (root/'fixture.html').write_text('''<!doctype html><canvas id="c" width="640" height="400"></canvas><script>
let frames=0,x=40,events=[];
addEventListener('keydown',e=>{events.push({key:e.key,trusted:e.isTrusted});if(e.key==='ArrowRight')x+=10});
function tick(){frames++;let c=document.getElementById('c').getContext('2d');c.fillStyle='#123';c.fillRect(0,0,640,400);c.fillStyle='#0f8';c.fillRect(x,100,40,40);requestAnimationFrame(tick)}
requestAnimationFrame(tick);
window.__JEV_STATE__=()=>({phase:'test',frames,x,events,time:performance.now()});
</script>''')
            host=BrowserHost('fixture.html',{'time':{'mode':'step'},'viewport':{'width':1280,'height':720}},
                             root,root/'artifacts',record=True,asset_root=root)
            try:
                host.start()
                before=host.snapshot()
                host.page.wait_for_timeout(100)
                self.assertEqual(host.snapshot(),before)
                host.keys(['ArrowRight'],10)
                after=host.snapshot()
                self.assertEqual(after['frames']-before['frames'],10)
                self.assertEqual(after['x']-before['x'],10)
                self.assertEqual(after['events'],[{'key':'ArrowRight','trusted':True}])
                self.assertGreater(len(host.screenshot()),1000)
            finally:
                host.close()
            video=root/'artifacts'/'full-run.webm'
            self.assertTrue(video.is_file())
            self.assertGreater(video.stat().st_size,1000)
            self.assertEqual(video.read_bytes()[:4],bytes.fromhex('1a45dfa3'))

    def test_native_time_continues_without_frame_advance_during_a_model_wait(self):
        import json
        with tempfile.TemporaryDirectory(prefix='jev-native-time-') as folder:
            root=Path(folder)
            (root/'fixture.html').write_text('''<!doctype html><script>
let frames=0,timed=false;
function tick(){frames++;requestAnimationFrame(tick)}requestAnimationFrame(tick);
setTimeout(()=>timed=true,100);
window.__JEV_STATE__=()=>({frames,timed,time:performance.now(),wall:Date.now(),virtualClock:!!window.__JEV_CLOCK__});
</script>''')
            host=BrowserHost('fixture.html',{'time':{'mode':'realtime'}},root,root/'artifacts',record=False,asset_root=root)
            try:
                host.start()
                # Load completion can precede the first compositor cycle. Require
                # native rAF startup, then measure a separate inference wait.
                host.frame.wait_for_function("window.__JEV_STATE__().frames >= 2", timeout=2000)
                before=host.snapshot()
                host.page.wait_for_timeout(200)
                after=host.snapshot()
                self.assertFalse(after['virtualClock'])
                self.assertTrue(after['timed'])
                self.assertGreater(after['frames'],before['frames'])
                self.assertGreater(after['time']-before['time'],150)
                self.assertGreater(after['wall']-before['wall'],150)
                manifest=json.loads((root/'artifacts/host.json').read_text())
                self.assertFalse(manifest['timing_contract']['game_paused_during_inference'])
                self.assertEqual(manifest['timing_contract']['mode'],'realtime')
            finally:host.close()

    def test_confirmation_deadline_freezes_during_model_thinking(self):
        with tempfile.TemporaryDirectory(prefix='jev-confirm-contract-') as folder:
            root=Path(folder)
            (root/'fixture.html').write_text('''<!doctype html><button id="confirm">Collect</button><script>
let pending=false,done=false,clicks=0;
document.getElementById('confirm').onclick=()=>{clicks++;if(!pending){pending=true;setTimeout(()=>pending=false,2400)}else{done=true}};
window.__JEV_STATE__=()=>({pending,done,clicks,time:performance.now()});
</script>''')
            host=BrowserHost('fixture.html',{'time':{'mode':'step'}},root,root/'artifacts',record=False,asset_root=root)
            try:
                host.start();host.click('#confirm');host.advance(60)
                before=host.snapshot();self.assertTrue(before['pending'])
                host.page.wait_for_timeout(2800)
                self.assertEqual(host.snapshot(),before)
                host.click('#confirm');host.advance(1)
                self.assertTrue(host.snapshot()['done'])
                self.assertEqual(host.snapshot()['clicks'],2)
            finally:host.close()

    def test_icon_controls_and_nested_prices_have_player_visible_labels(self):
        with tempfile.TemporaryDirectory(prefix='jev-dom-label-contract-') as folder:
            root=Path(folder)
            (root/'fixture.html').write_text('''<!doctype html>
<button onclick="window.clicks++"><img alt="烈性合剂" width="24" height="32" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E">Q</button>
<div><h3>防御护符：获得格挡</h3><div><span><button onclick="window.clicks++">120</button></span></div></div>
<div id="target" class="cursor-crosshair" style="cursor:crosshair">敌人 24/24</div><div class="cursor-grab" style="cursor:grab">拖拽容器</div><div class="cursor-default" style="cursor:default">装饰容器</div><div>其它界面文字 <button>结束回合</button></div>
<script>window.clicks=0;document.getElementById('target').addEventListener('click',()=>window.clicks++);window.__JEV_STATE__=()=>window.__JEV_DOM__.snapshot(document);</script>''')
            host=BrowserHost('fixture.html',{'time':{'mode':'step'}},root,root/'artifacts',record=False,asset_root=root)
            try:
                host.start();actions=host.snapshot()['candidates']
                self.assertEqual(len(actions),4)
                potion=next(item for item in actions if '烈性合剂' in item['label'])
                price=next(item for item in actions if '120' in item['label'])
                self.assertIn('Q',potion['label'])
                self.assertIn('防御护符',price['label'])
                self.assertIn('结束回合',[item['label'] for item in actions])
                self.assertFalse(any('容器' in item['label'] for item in actions))
                host.click(price['selector'])
                self.assertEqual(host.frame.evaluate('window.clicks'),1)
                target=next(item for item in actions if '敌人 24/24' in item['label'])
                host.click(target['selector'])
                self.assertEqual(host.frame.evaluate('window.clicks'),2)
            finally:host.close()

    def test_declared_semantics_expose_an_existing_overlay_without_activating_it(self):
        with tempfile.TemporaryDirectory(prefix='jev-overlay-contract-') as folder:
            root=Path(folder)
            source='''<!doctype html><div id="overlay" onclick="window.clicks++" style="position:fixed;inset:0">整塔全览：点击任意处返回</div>
<script>window.clicks=0;window.__JEV_STATE__=()=>window.__JEV_DOM__.snapshot(document);</script>'''
            (root/'fixture.html').write_text(source)
            config={'time':{'mode':'step'},'dom_semantics':[{'selector':'#overlay','role':'button','label':'关闭整塔全览'}]}
            host=BrowserHost('fixture.html',config,root,root/'artifacts',record=False,asset_root=root)
            try:
                host.start();actions=host.snapshot()['candidates']
                self.assertEqual(len(actions),1)
                self.assertTrue(actions[0]['label'].startswith('关闭整塔全览'))
                self.assertEqual(host.frame.evaluate('window.clicks'),0)
                host.click(actions[0]['selector'])
                self.assertEqual(host.frame.evaluate('window.clicks'),1)
                self.assertEqual((root/'fixture.html').read_text(),source)
            finally:host.close()

    def test_visible_resource_fields_keep_zero_distinct_from_missing(self):
        with tempfile.TemporaryDirectory(prefix='jev-resource-contract-') as folder:
            root=Path(folder)
            (root/'fixture.html').write_text('''<!doctype html>
<span id="energy">3</span><span id="max">/3</span><span id="hidden" style="display:none">999</span>
<button onclick="document.getElementById('energy').textContent='0';window.clicks++">Spend</button>
<script>window.clicks=0;window.__JEV_STATE__=()=>window.__JEV_DOM__.snapshot(document);</script>''')
            fields={'energy_current':{'selector':'#energy','type':'number'},
                    'energy_max':{'selector':'#max','pattern':r'(\d+)','type':'number'},
                    'unseen':{'selector':'#hidden','type':'number'},
                    'missing':{'selector':'#absent','type':'number'}}
            host=BrowserHost('fixture.html',{'time':{'mode':'step'},'dom_fields':fields},root,root/'artifacts',record=False,asset_root=root)
            try:
                host.start();before=host.snapshot()
                self.assertEqual(before['fields'],{'energy_current':3,'energy_max':3})
                self.assertEqual(host.frame.evaluate('window.clicks'),0)
                host.click('button');after=host.snapshot()
                self.assertEqual(after['fields'],{'energy_current':0,'energy_max':3})
                self.assertNotEqual(before['signature'],after['signature'])
                self.assertEqual(host.frame.evaluate('window.clicks'),1)
            finally:host.close()

    def test_terminal_animation_finishes_without_an_extra_game_action(self):
        from general_agent.adapters.web_runtime import WebRuntimeAdapter
        with tempfile.TemporaryDirectory(prefix='jev-terminal-contract-') as folder:
            root=Path(folder)
            (root/'fixture.html').write_text('''<!doctype html><button id="win">Win</button><div id="score">0</div><script>
let won=false,score=0,clicks=0;
document.getElementById('win').onclick=()=>{clicks++;won=true;setTimeout(()=>{score=100;document.getElementById('score').textContent=score},1200)};
window.__JEV_STATE__=()=>({phase:won?'victory':'playing',done:won,success:won,score,clicks,time:performance.now()});
</script>''')
            config={'time':{'mode':'step','terminal_frames':90}}
            adapter=WebRuntimeAdapter('test',config=config)
            adapter.artifacts=root/'artifacts'
            adapter.host=BrowserHost('fixture.html',config,root,adapter.artifacts,record=False,asset_root=root)
            try:
                adapter.host.start();adapter.host.click('#win')
                self.assertEqual(adapter.host.snapshot()['score'],0)
                obs=adapter.observe()
                self.assertTrue(obs.done and obs.success)
                self.assertEqual(adapter._state['score'],100)
                self.assertEqual(adapter._state['clicks'],1)
                self.assertEqual(adapter.steps,0)
                self.assertEqual(obs.raw['terminal_frames'],90)
                before=adapter.host.snapshot()
                adapter.observe()
                self.assertEqual(adapter.host.snapshot(),before)
                self.assertTrue((adapter.artifacts/'final.png').is_file())
            finally:adapter.close()
