"""Run self-contained model episodes with raw browser videos and audit logs."""
from __future__ import annotations
import argparse
import json
import signal
from pathlib import Path
import time

from .adapters.web_runtime import WebRuntimeAdapter
from .live import LiveRun
from webenv.timing import timing_contract


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game',action='append',required=True,help='Catalog ID; repeat for a sequential suite')
    parser.add_argument('--goal',default='')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--time-mode',choices=['step','realtime'],help='step freezes inference; realtime uses native browser time (experimental controller)')
    parser.add_argument('--fast-context',choices=['standard','compact','criteria'],help='Opt-in compact fast prompt; slow Thread context remains complete')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--max-seconds',type=float)
    parser.add_argument('--diagnostic',action='store_true',help='Keep recording but never count this run as acceptance')
    parser.add_argument('--no-record',action='store_true',help='Diagnostic run only; not video acceptance')
    parser.add_argument('--no-distill',action='store_true',help='Skip the slow-model postmortem (diagnostics or externally coordinated reviews)')
    parser.add_argument('--game-only-distill',action='store_true',help='Persist only this title lessons; defer common memory updates')
    parser.add_argument('--continue-on-failure',action='store_true')
    args=parser.parse_args()
    failures=0
    active_run=None
    stop_requested=False
    def stop_signal(signum,frame):
        nonlocal stop_requested
        stop_requested=True
        if active_run is not None:active_run.stop()
    signal.signal(signal.SIGINT,stop_signal)
    signal.signal(signal.SIGTERM,stop_signal)
    for game in args.game:
        run=LiveRun(game,args.goal)
        active_run=run
        if not isinstance(run.adapter,WebRuntimeAdapter):
            parser.error(f'{game} has no knowledge/runtime.json; this CLI only runs Web Runtime packs')
        if args.time_mode is not None:
            run.adapter.config['time']={**run.adapter.config.get('time',{}),'mode':args.time_mode}
        if args.fast_context is not None:
            run.adapter.config['fast_context_mode']=args.fast_context
            run.policy.context_mode=args.fast_context
            if args.fast_context=='criteria':
                run.adapter.config['slow_action_criteria']=True
                run.outer.generate_criteria=True
        timing=timing_contract(run.adapter.config)
        run.adapter.config['seed']=args.seed
        run.adapter.config['record_video']=not args.no_record
        if args.max_steps is not None:run.adapter.config['max_steps']=args.max_steps
        if args.max_seconds is not None:run.adapter.config['max_seconds']=args.max_seconds
        source_paths=['general_agent/progress.py','general_agent/monitor.py','general_agent/scene_memory.py','general_agent/recurrence.py','general_agent/context.py','general_agent/runtime_cli.py','general_agent/learning.py',
                      'harness.py','experience.py','general_agent/types.py','general_agent/policy.py','general_agent/outer.py','general_agent/runner.py',
                      'general_agent/live.py','general_agent/adapters/web_runtime.py',
                      'webenv/host.py','webenv/timing.py','webenv/clock.js','webenv/dom.js','webenv/dom.py','webenv/navigation.py']
        root=Path(__file__).resolve().parents[1]
        pack=root/'knowledge'/game
        import hashlib
        hashes={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in source_paths}
        pack_sources=[pack/'runtime.json',pack/'runtime_hook.js',pack/'experience.json',root/'knowledge/common/web-runtime.experience.json']
        pack_sources += [pack/hook['script'] for hook in run.adapter.config.get('source_hooks',[])]
        for path in pack_sources:
            if path.is_file():hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, expected in hashes.items():
            data=(root/relative).read_bytes()
            if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('Source changed during snapshot: '+relative)
            target=run.folder/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        import harness
        review_generation=run.adapter.config.get('review_generation',run.adapter.config.get('slow_generation',{}))
        (run.folder/'protocol.json').write_text(json.dumps({
            'trial_type':'diagnostic' if args.no_record or args.diagnostic else 'acceptance_fresh_start',
            'game':game,'environment_seed':args.seed,'root_goal':run.goal,'timing_contract':timing,
            'initial_state':'fresh isolated browser context; no imported save','runtime_config':run.adapter.config,
            'model_configuration':{'slow_model':run.outer.model,
                'outer_generation':harness.chat_generation(**run.outer.generation),
                'review_generation':harness.chat_generation(**review_generation)},
            'assistant_gameplay_actions':0,'record_video':not args.no_record,'source_sha256':hashes,'source_snapshot':'source/'},ensure_ascii=False,indent=2))
        run.start()
        print(json.dumps({'event':'started','game':game,'seed':args.seed,'run':str(run.folder),
                          'video':not args.no_record,'timing_contract':timing},ensure_ascii=False),flush=True)
        last_report=0.0
        while run.running:
            time.sleep(1)
            if time.monotonic()-last_report<10:continue
            last_report=time.monotonic()
            snapshot=run.snapshot()
            print(json.dumps({'event':'progress','game':game,'step':snapshot['steps'],
                'status':snapshot['status'],'choice':snapshot['choice'],
                'jev':snapshot['model'],'deepseek':snapshot['outer_model'],
                'outer_calls':snapshot['outer_calls'],'elapsed':snapshot['elapsed']},ensure_ascii=False),flush=True)
        result=run.result
        success=bool(result and result.success)
        print(json.dumps({'event':'finished','game':game,'success':success,'run':str(run.folder),
             'steps':result.steps if result else run.steps,'error':run.error,
             'video':str(run.folder/'browser'/'full-run.webm') if not args.no_record else None},ensure_ascii=False),flush=True)
        if not stop_requested and not args.no_distill and (run.folder/'trace.jsonl').is_file():
            try:
                from .learning import distill
                report=distill(run.folder,persist_scopes={'game'} if args.game_only_distill else None)
                print(json.dumps({'event':'experience_saved','game':game,'model':report['author_model'],
                                  'facts':len(report.get('accepted_facts',[]))},ensure_ascii=False),flush=True)
            except Exception as exc:
                print(json.dumps({'event':'reflection_error','game':game,'error':str(exc)},ensure_ascii=False),flush=True)
        if stop_requested:return 130
        if not success:
            failures+=1
            if not args.continue_on_failure:break
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
