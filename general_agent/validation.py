"""Build acceptance status from actual terminal, model, protocol and video evidence."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]



def episode_metrics(summary: dict, rows: list) -> dict:
    """Report recorded costs and input effects, without treating change as progress."""
    decisions=[row.get('decision') or {} for row in rows]
    outer=[call for row in rows for call in row.get('outer_calls',[])]
    def latency(calls):
        samples=sorted(call['ms'] for call in calls
                       if isinstance(call.get('ms'),(int,float)) and call['ms']>0)
        return {'recorded_samples':len(samples),'total_ms':round(sum(samples)),
                'median_ms':statistics.median(samples) if samples else None,
                'p95_ms':samples[max(0,math.ceil(len(samples)*.95)-1)] if samples else None}
    executions=summary.get('executions') or []
    # The summary owns the executed-input list, including the final action.
    # Trace rows instead acknowledge the preceding action and omit the final one.
    effects={name:sum(item.get(name) is True for item in executions) for name in ['ok','changed']}
    return {'decision_sources':dict(Counter(item.get('by','unknown') for item in decisions)),
            'jev_calls':sum(bool(item.get('model')) for item in decisions),
            'deepseek_calls':len(outer),'deepseek_failures':sum(bool(call.get('error')) for call in outer),
            'jev_latency':latency(decisions),'deepseek_latency':latency(outer),
            'executions':{'count':len(executions),'acknowledged':effects['ok'],
                          'state_changed':effects['changed'],
                          'note':'A changed observation is an input effect, not proof of game progress.'}}


def assess(folder: Path, required_time_mode=None):
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8')) if (folder/'protocol.json').is_file() else {}
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8')) if (folder/'summary.json').is_file() else {
        'game':protocol.get('game'),'run_id':folder.name,'result':{},
        'error':protocol.get('interruption','run has no finalized result')}
    result=summary.get('result') or {}
    trace=folder/'trace.jsonl'
    rows=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()] if trace.is_file() else []
    jev={row.get('decision',{}).get('model','') for row in rows}
    outer={call.get('model','') for row in rows for call in row.get('outer_calls',[])}
    video=folder/'browser/full-run.webm';final=folder/'browser/final.png'
    host=folder/'browser/host.json'
    host_data=json.loads(host.read_text(encoding='utf-8')) if host.is_file() else {}
    video_valid=False
    if video.is_file() and video.stat().st_size>1000:
        with video.open('rb') as handle:video_valid=handle.read(4)==bytes.fromhex('1a45dfa3')
    recorded_timing=protocol.get('timing_contract') or host_data.get('timing_contract') or {}
    mode=recorded_timing.get('mode') or (protocol.get('runtime_config',{}).get('time') or {}).get('mode') or (host_data.get('clock') or {}).get('time_mode') or 'unknown'
    timing={'mode':mode,'validation_class':'stepped_feasibility' if mode=='step' else 'realtime' if mode=='realtime' else 'unknown',
            'game_paused_during_inference':True if mode=='step' else False if mode=='realtime' else None}
    checks={
      'terminal_success':result.get('success') is True and result.get('stopped') is not True,
      'fresh_start':protocol.get('trial_type')=='acceptance_fresh_start',
      'assistant_gameplay_zero':protocol.get('assistant_gameplay_actions')==0,
      'jev_called':any('jev' in model.lower() for model in jev),
      'deepseek_called':any('deepseek' in model.lower() for model in outer),
      'recording_enabled':protocol.get('record_video') is True and host_data.get('recording') is True,
      'video_finalized':video_valid,
      'final_image':final.is_file(),
    }
    base_accepted=all(checks.values())
    if required_time_mode is not None:checks['required_time_mode']=mode==required_time_mode
    return {'game':summary['game'],'run_id':summary['run_id'],'accepted':all(checks.values()),
      'base_accepted':base_accepted,'timing':timing,'required_time_mode':required_time_mode,
      'checks':checks,'success':result.get('success') is True,'steps':result.get('steps'),
      'elapsed':result.get('seconds'),'error':summary.get('error'),
      'models':{'jev':sorted(jev-{''}),'deepseek':sorted(outer-{''})},
      'protocol':protocol.get('trial_type'),'run_folder':str(folder),
      'metrics':episode_metrics(summary,rows),
      'video':str(video) if video.is_file() else None,'final_image':str(final) if final.is_file() else None}


def build(suite: Path, runs: Path, required_time_mode=None):
    config=json.loads(suite.read_text(encoding='utf-8'));groups={item['id']:[] for item in config['games']}
    folders={file.parent for pattern in ['*/summary.json','*/protocol.json'] for file in runs.glob(pattern)}
    def finalized_at(folder):
        artifact=folder/('summary.json' if (folder/'summary.json').is_file() else 'protocol.json')
        return (artifact.stat().st_mtime_ns,folder.name)
    folders=sorted(folders,key=finalized_at)
    for folder in folders:
        file=folder/'summary.json' if (folder/'summary.json').is_file() else folder/'protocol.json'
        data=json.loads(file.read_text(encoding='utf-8'))
        if data.get('game') in groups:groups[data['game']].append(assess(folder,required_time_mode))
    games=[]
    for item in config['games']:
        attempts=groups[item['id']];accepted=[attempt for attempt in attempts if attempt['accepted']]
        games.append({**item,'accepted':bool(accepted),'accepted_run':accepted[-1] if accepted else None,
          'stepped_feasibility_accepted':any(trial['base_accepted'] and trial['timing']['mode']=='step' for trial in attempts),
          'realtime_accepted':any(trial['base_accepted'] and trial['timing']['mode']=='realtime' for trial in attempts),'attempts':attempts})
    return {'suite':config['name'],'required_time_mode':required_time_mode,'all_accepted':all(game['accepted'] for game in games),
            'all_realtime_accepted':all(game['realtime_accepted'] for game in games),'games':games}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,required=True);parser.add_argument('--runs',type=Path,default=ROOT/'runs/panel')
    parser.add_argument('--require-time-mode',choices=['step','realtime'],help='Reject clears recorded with another clock')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/validation/report.json');args=parser.parse_args()
    report=build(args.suite,args.runs,args.require_time_mode);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'all_accepted':report['all_accepted'],'required_time_mode':args.require_time_mode,'all_realtime_accepted':report['all_realtime_accepted'],
      'games':[{'id':game['id'],'accepted':game['accepted'],'realtime_accepted':game['realtime_accepted']} for game in report['games']],
      'report':str(args.output)},ensure_ascii=False))

if __name__=='__main__':main()
