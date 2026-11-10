import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from general_agent.adapters.web_runtime import WebRuntimeAdapter
from webenv.host import BrowserHost

ROOT=Path(__file__).resolve().parents[2]

class RuntimeContracts(unittest.TestCase):
    def test_control_annotations_preserve_choices_and_only_add_public_semantics(self):
        from webenv.dom import annotate_candidates
        original=[{'id':'enemy','label':'烬小鬼 10/27'}, {'id':'card','label':'打击 攻击 造成6伤害'}]
        rules=[{'flag':'selecting','pattern':r'\d+/\d+','prefix':'确认目标：','role':'target'},
               {'flag':'selecting','pattern':'攻击','prefix':'切换/取消：','role':'card_selection'}]
        annotated=annotate_candidates(original,{'selecting':True},rules)
        self.assertEqual([item['id'] for item in annotated],['enemy','card'])
        self.assertEqual(annotated[0]['role'],'target')
        self.assertEqual(annotated[1]['role'],'card_selection')
        self.assertNotIn('role',original[0])
        self.assertEqual(annotate_candidates(original,{'selecting':False},rules),original)

    def test_internal_autoplayer_is_rejected_before_decision(self):
        from types import SimpleNamespace
        adapter=WebRuntimeAdapter('test',config={'integrity':{'built_in_ai':False}})
        adapter.host=SimpleNamespace(snapshot=lambda:{'integrity':{'built_in_ai':True}})
        with self.assertRaisesRegex(RuntimeError,'integrity failed'):
            adapter._read()

    def test_target_paging_keeps_every_legal_target_available(self):
        from types import SimpleNamespace
        keys={'up':'ArrowUp','down':'ArrowDown','left':'ArrowLeft','right':'ArrowRight'}
        state={'phase':'play','navigation':{'position':[0,1],'grid':['x'*40,'.'*40],
               'walkable':['.'],'targets':[{'id':'t'+str(i),'x':i,'y':0,'label':'target'+str(i)} for i in range(40)]}}
        adapter=WebRuntimeAdapter('test',config={'navigation':{'keys':keys}})
        adapter.host=SimpleNamespace(snapshot=lambda:state)
        first=adapter._actions(state)
        self.assertEqual(len(first),32)
        adapter.act(next(action for action in first if action.id=='next_targets'))
        second=adapter._actions(state)
        ids={action.id for action in first+second if action.id!='next_targets'}
        self.assertEqual(ids,{'t'+str(i) for i in range(40)})

    def test_ambiguous_source_hook_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'game.js').write_text('ANCHOR ANCHOR')
            (root/'hook.js').write_text('readonly hook')
            host=BrowserHost('game.js',{'source_hooks':[{'file':'game.js','anchor':'ANCHOR','script':'hook.js'}]},
                             root,root/'artifacts',asset_root=root,record=False)
            with self.assertRaisesRegex(ValueError,'match once'):
                host._asset('game.js')
            self.assertEqual((root/'game.js').read_text(),'ANCHOR ANCHOR')

    def test_mota_projection_does_not_reveal_future_floors_or_mutate_game(self):
        hook=(ROOT/'knowledge/kimi-1bbf/runtime_hook.js').read_text()
        script='''const vm=require('node:vm');
const game={f:0,x:1,y:1,hp:1000,atk:10,def:10,gold:0,keys:[0,0,0],
 floors:[['#####'.split(''),'#..1#'.split(''),'#####'.split('')],['SECRET_FUTURE_FLOOR'.split('')]]};
const original=JSON.stringify(game);
const context={game,screen:'play',document:{body:{innerText:'visible HUD'}},DATA:{MONSTERS:{},ITEM_TEXT:{'1':'黄钥匙'},FLOORS:[{},{}]},CORE:{},window:{}};
vm.createContext(context);vm.runInContext(HOOK,context);
const snapshot=JSON.stringify(context.window.__JEV_STATE__());
if(snapshot.includes('SECRET'))throw Error('Future-floor leak');
if(JSON.stringify(game)!==original)throw Error('Projection mutated game');
console.log(snapshot);'''.replace('HOOK',json.dumps(hook))
        run=subprocess.run(['node','-e',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)
        state=json.loads(run.stdout)
        self.assertEqual(state['floor'],1)
        self.assertEqual(len(state['navigation']['targets']),1)

    def test_visible_boss_is_not_lost_to_entity_cap_and_hidden_coordinates_stay_private(self):
        hook=(ROOT/'knowledge/vampire-survivors-9256/runtime_hook.js').read_text()
        script="""const vm=require('node:vm');
const player={x:100,y:200,hp:100,maxHp:100,speed:210,pickup:100,level:1,xp:0,xpNext:10,iframes:0,weapons:[]};
const boss={x:500,y:300,r:30,isBoss:true,hp:90,maxHp:100};
const enemies=Array.from({length:30},(_,i)=>({x:110+i,y:200,r:10,hp:10,maxHp:10,type:'ghost'}));
enemies.push(boss);
const context={window:{},state:'playing',player,enemies,aiEnabled:false,envMoveVec:null,
 W:1000,H:600,gameTime:500,kills:10,foes:[],gems:[],items:[],chests:[],altars:[],
 document:{body:{innerText:''},getElementById:()=>({innerText:''})}};
vm.createContext(context);vm.runInContext(HOOK,context);
const before=JSON.stringify({player,enemies});const visible=context.window.__JEV_STATE__();
if(JSON.stringify({player,enemies})!==before)throw Error('Projection mutated original entities');
boss.x=5000;const hidden=context.window.__JEV_STATE__();
console.log(JSON.stringify({visible,hidden}));""".replace('HOOK',json.dumps(hook))
        run=subprocess.run(['node','-e',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)
        states=json.loads(run.stdout);visible=states['visible'];hidden=states['hidden']
        self.assertEqual(len(visible['enemies']),24)
        self.assertFalse(any(enemy['boss'] for enemy in visible['enemies']))
        self.assertEqual([visible['boss']['dx'],visible['boss']['dy']],[400,100])
        self.assertTrue(visible['boss']['visible'])
        self.assertFalse(hidden['boss']['visible'])
        self.assertNotIn('dx',hidden['boss']);self.assertNotIn('dy',hidden['boss'])
        self.assertEqual(hidden['boss']['direction_x'],1)

    def test_tennis_projection_excludes_opponent_private_policy(self):
        hook=(ROOT/'knowledge/kk-fcd4/runtime_hook.js').read_text()
        script='''const vm=require('node:vm');
const context={window:{},ST:'play',matchWinner:null,DEMO:false,SPEED:1,Q:new URLSearchParams(),
 DIFFS:[{name:'NORMAL'}],diff:0,W:960,H:600,NETX:480,SY0:152,SY1:448,CX0:120,CX1:840,
 kk:{x:210,y:300,swingCd:0,swing:0},bot:{x:750,y:300,ai:{private_secret:'HIDDEN_AI_TARGET'}},
 games:{kk:0,bot:0},pts:{kk:0,bot:0},server:'kk',pointBanner:'',
 ball:{x:100,y:200,z:20,vx:0,vy:0,vz:0,live:true,serving:true,tossed:false,lastHit:null,serveAim:null}};
vm.createContext(context);vm.runInContext(HOOK,context);
console.log(JSON.stringify(context.window.__JEV_STATE__()));'''.replace('HOOK',json.dumps(hook))
        run=subprocess.run(['node','-e',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertNotIn('HIDDEN_AI_TARGET',run.stdout)
        self.assertEqual(json.loads(run.stdout)['opponent'],{'x':750,'y':300})

    def test_tennis_landing_marker_is_projected_only_when_original_ui_displays_it(self):
        hook=(ROOT/'knowledge/kk-fcd4/runtime_hook.js').read_text()
        script='''const vm=require('node:vm');
let predictions=0;
const ball={x:600,y:200,z:20,vx:-100,vy:0,vz:0,live:true,serving:true,tossed:false,lastHit:'bot',firstBounceDone:false,serveAim:null};
const context={window:{},ST:'play',matchWinner:null,DEMO:false,SPEED:1,Q:new URLSearchParams(),
 DIFFS:[{name:'NORMAL'}],diff:0,W:960,H:600,NETX:480,SY0:152,SY1:448,CX0:120,CX1:840,
 kk:{x:210,y:300,swingCd:0,swing:0},bot:{x:750,y:300,ai:{secret:'PRIVATE'}},
 games:{kk:0,bot:0},pts:{kk:0,bot:0},server:'bot',pointBanner:'',ball,
 predictLanding:()=>{predictions++;return{x:200,y:300}}};
vm.createContext(context);vm.runInContext(HOOK,context);
const read=()=>context.window.__JEV_STATE__().ball.visible_landing_marker;
if(read()!==null||predictions!==0)throw Error('Prediction exposed during serve');
ball.serving=false;
if(JSON.stringify(read())!==JSON.stringify({x:200,y:300})||predictions!==1)throw Error('Visible circle missing');
ball.firstBounceDone=true;
if(read()!==null||predictions!==1)throw Error('Prediction exposed after circle disappears');
console.log('visible-only');'''.replace('HOOK',json.dumps(hook))
        run=subprocess.run(['node','-e',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_mota_current_stairs_are_exposed_without_mutating_the_floor(self):
        hook=(ROOT/'knowledge/kimi-1bbf/runtime_hook.js').read_text()
        script='''const vm=require('node:vm');
const game={f:1,x:1,y:1,hp:1000,atk:10,def:10,gold:0,keys:[0,0,0],
 floors:[['PAST'.split('')],['###'.split(''),'#>.'.split(''),'###'.split('')],['SECRET_FUTURE'.split('')]]};
const original=JSON.stringify(game);
const context={game,screen:'play',document:{body:{innerText:'floor 2'}},DATA:{MONSTERS:{},ITEM_TEXT:{},FLOORS:[{},{},{}]},CORE:{},window:{}};
vm.createContext(context);vm.runInContext(HOOK,context);
const value=context.window.__JEV_STATE__();
if(JSON.stringify(game)!==original)throw Error('Mutated floor');
console.log(JSON.stringify(value.navigation));'''.replace('HOOK',json.dumps(hook))
        run=subprocess.run(['node','-e',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)
        navigation=json.loads(run.stdout)
        self.assertNotIn('SECRET_FUTURE',run.stdout)
        self.assertEqual(navigation['targets'][0]['label'],'下楼至第1层')
        self.assertTrue(navigation['targets'][0]['available'])
        self.assertEqual([navigation['targets'][0]['x'],navigation['targets'][0]['y']],[1,1])


    def test_key_candidates_report_actual_simulation_duration_to_models(self):
        adapter=WebRuntimeAdapter('test',config={'time':{'dt_ms':1000/60},
            'controls':[{'id':'move','label':'move','keys':['ArrowRight'],'frames':12}]})
        action=adapter._actions({'phase':'play'})[0]
        self.assertAlmostEqual(action.seconds,.2)
        self.assertEqual(action.params['frames'],12)
        self.assertIn('hold 0.2s',action.describe())

    def test_fast_state_includes_the_same_declared_control_contract(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        adapter=WebRuntimeAdapter('test',config={'instructions':'A key triggers once at the start.',
            'fast_fields':['phase'],'controls':[]})
        adapter.host=SimpleNamespace(snapshot=lambda:{'phase':'play'},screenshot=lambda:b'preview')
        with patch('general_agent.adapters.web_runtime.render_lessons',return_value=''):
            observation=adapter.observe()
        self.assertEqual(observation.state['fast_state']['control_instructions'],
                         observation.state['instructions'])
