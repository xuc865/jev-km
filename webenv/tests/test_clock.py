import json
from pathlib import Path
import subprocess
import unittest

CLOCK=Path(__file__).resolve().parents[1]/'clock.js'

class ClockTests(unittest.TestCase):
    def test_deterministic_steps_timers_and_frozen_inference(self):
        source=CLOCK.read_text()
        script='''const vm=require('node:vm');
async function trial(){
 const sandbox={location:{pathname:'/fixture/game'},performance:{now:()=>900},Date,Math:Object.create(Math),
  setTimeout,clearTimeout,requestAnimationFrame:fn=>setTimeout(()=>fn(0),0),console};
 sandbox.window=sandbox;sandbox.__JEV_BOOT__={time_mode:'step',seed:23};
 vm.createContext(sandbox);vm.runInContext(SOURCE,sandbox);
 vm.runInContext(`globalThis.count=0;globalThis.ticks=[];globalThis.random=[];
 function loop(){count++;random.push(Math.random());requestAnimationFrame(loop)}
 requestAnimationFrame(loop);setTimeout(()=>ticks.push(Date.now()),30);
 globalThis.cancelled=false;const old=setTimeout(()=>cancelled=true,10);clearTimeout(old);`,sandbox);
 await new Promise(resolve=>setTimeout(resolve,25));
 if(sandbox.count!==0)throw Error('World advanced while model was thinking');
 await sandbox.__JEV_CLOCK__.advanceFrames(100,false);
 return {count:sandbox.count,ticks:sandbox.ticks,random:sandbox.random,cancelled:sandbox.cancelled,time:sandbox.performance.now()};
}
(async()=>{const a=await trial(),b=await trial();
 if(JSON.stringify(a)!==JSON.stringify(b))throw Error('Non-deterministic identical trials');
 if(a.count!==100||a.ticks.length!==1||a.cancelled||Math.abs(a.time-1000/60*100)>1e-7)throw Error('Wrong clock results');
 console.log(JSON.stringify({frames:a.count,deterministic:true,frozen:true}));})();'''
        script=script.replace('SOURCE',json.dumps(source))
        result=subprocess.run(['node','-e',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout)['frozen'])
