"""Execute the shipped panel command bridge against input/transport fixtures."""
import json
from pathlib import Path
import subprocess
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'web/panel_bridge.js'


class PanelBridgeTests(unittest.TestCase):
    def run_js(self, body):
        setup = """const vm=require('node:vm');
const context={window:{}};vm.createContext(context);vm.runInContext(SOURCE,context);
const Bridge=context.window.__JEV_PANEL_BRIDGE__;
let clicks=0,events=[];
const el={disabled:false,getAttribute:()=>null,click:()=>clicks++,dispatchEvent:e=>events.push(e)};
const doc={querySelector:()=>el,elementFromPoint:()=>el,
 defaultView:{MouseEvent:class {constructor(type,props){this.type=type;Object.assign(this,props)}}}};
const state={run_id:'run',game:'fixture',pending_click:{command_id:'cmd',selector:'#button',signature:'current'}};
const snapshot=()=>({signature:'current',text:'public result'});
""".replace('SOURCE', json.dumps(SOURCE.read_text()))
        result=subprocess.run(['node','-e',setup+'\n(async()=>{'+body+'})().catch(e=>{console.error(e);process.exitCode=1});'],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_failed_frame_or_ack_post_retries_ack_without_repeating_input(self):
        self.run_js("""
for(const endpoint of ['/api/live/frame','/api/live/clicked']){
 clicks=0;let failed=false,posts=[];
 const bridge=new Bridge({snapshot,send:async(path,payload)=>{posts.push({path,payload});if(path===endpoint&&!failed){failed=true;throw Error('temporary POST failure')}}});
 let rejected=false;try{await bridge.apply(doc,state)}catch{rejected=true}
 if(!rejected||bridge.active||clicks!==1)throw Error('Failure did not release the bridge');
 await bridge.apply(doc,state);const count=posts.length;await bridge.apply(doc,state);
 if(clicks!==1||posts.length!==count||!posts.at(-1).payload.ok)throw Error('Retry repeated input or lost acknowledgement');
}
""")

    def test_stale_command_is_acknowledged_without_clicking(self):
        self.run_js("""
const posts=[];const bridge=new Bridge({snapshot,send:async(path,payload)=>posts.push({path,payload})});
await bridge.apply(doc,{...state,pending_click:{...state.pending_click,signature:'old'}});
if(clicks||posts.at(-1).payload.ok||!posts.at(-1).payload.error.includes('stale'))throw Error('Stale command executed');
""")

    def test_legacy_coordinate_command_preserves_mouse_coordinates(self):
        self.run_js("""
const bridge=new Bridge({snapshot,send:async()=>{}});
await bridge.apply(doc,{...state,pending_click:{command_id:'legacy',x:42,y:73}});
if(clicks||events.length!==1||events[0].clientX!==42||events[0].clientY!==73)throw Error('Legacy coordinate input changed');
""")
