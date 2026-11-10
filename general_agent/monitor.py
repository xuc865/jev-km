"""Read-only monitoring of independently recorded Runtime episodes.

Trace freshness is reported as evidence, never as proof that a process is alive.
This module cannot send game inputs or control an episode.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
from webenv.timing import timing_contract


def trace_tail(path: Path, budget=524288):
    if not path.is_file():return []
    with path.open('rb') as handle:
        handle.seek(max(0,path.stat().st_size-budget))
        lines=handle.read().splitlines()
    rows=[]
    for line in reversed(lines):
        try:row=json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
        if isinstance(row,dict) and 'decision' in row:rows.append(row)
        if len(rows)>=8:break
    return list(reversed(rows))


class EpisodeMonitor:
    def __init__(self, root: Path):
        self.root=root
        self.cache={}

    def _snapshot(self, folder):
        trace=folder/'trace.jsonl';summary=folder/'summary.json';protocol=folder/'protocol.json'
        stamp=tuple(path.stat().st_mtime_ns if path.is_file() else 0 for path in [trace,summary,protocol])
        cached=self.cache.get(folder.name)
        if cached and cached[0]==stamp:return cached[1]
        rows=trace_tail(trace)
        if not rows:return None
        row=rows[-1];obs=row.get('observation') or {};state=obs.get('state') or {};decision=row['decision']
        protocol_data=json.loads(protocol.read_text(encoding='utf-8')) if protocol.is_file() else {}
        if protocol_data.get('trial_type')!='acceptance_fresh_start':return None
        result=json.loads(summary.read_text(encoding='utf-8')) if summary.is_file() else {}
        final=result.get('result') or {};outer={};outer_model=''
        for item in reversed(rows):
            for call in reversed(item.get('outer_calls',[])):
                try:outer=json.loads(call.get('response') or '')
                except (ValueError,TypeError):continue
                if isinstance(outer,dict):
                    outer_model=call.get('model','');break
                outer={}
            if outer:break
        snapshot={'game':protocol_data.get('game',result.get('game','')),'run_id':folder.name,
            'timing_contract':timing_contract(protocol_data['runtime_config']) if 'runtime_config' in protocol_data else None,
            'run_folder':str(folder),'external_snapshot':True,'kind':'external','running':False,
            'steps':final.get('steps',row.get('step',0)+1),'level':obs.get('level'),
            'elapsed':final.get('seconds') or row.get('elapsed',0),'choice':decision.get('action_id',''),
            'model':decision.get('model',''),'by':decision.get('by',''),'ms':decision.get('ms',0),
            'image_count':decision.get('image_count',0),'probabilities':decision.get('probabilities',{}),
            'actions':obs.get('actions',[]),'recent':[{'by':r['decision'].get('by'),'choice':r['decision'].get('action_id'),'ms':r['decision'].get('ms')} for r in rows],
            'outer_model':outer_model,'briefing':outer.get('briefing',''),'stage_goal':outer.get('stage_goal',''),
            'play_instruction':outer.get('play_instruction',''),'error':result.get('error',''),
            'public_state':state.get('runtime') or state,'finalized':summary.is_file(),
            'success':final.get('success') is True and final.get('stopped') is not True,
            'trace_at':trace.stat().st_mtime,'summary_at':summary.stat().st_mtime if summary.is_file() else 0}
        self.cache[folder.name]=(stamp,snapshot)
        return snapshot

    def snapshots(self, now=None):
        now=time.time() if now is None else now
        folders=[path.parent for path in self.root.glob('*/protocol.json')]
        def activity(folder):
            path=folder/'trace.jsonl'
            return path.stat().st_mtime if path.is_file() else 0
        folders.sort(key=activity,reverse=True)
        rows=[];completed=0
        for folder in folders[:24]:
            try:item=self._snapshot(folder)
            except (OSError,ValueError,KeyError):continue
            if not item:continue
            item=dict(item);age=max(0,round(now-item.pop('trace_at')))
            if item['finalized']:
                if completed>=4:continue
                completed+=1
                item['status']='胜利已归档' if item['success'] else '已结束 · 未通关'
            else:
                if age>600:continue
                item['status']='最近有决策 · 只读监控' if age<180 else '等待新记录 · 进程状态待核对'
            item['trace_age_seconds']=age
            rows.append(item)
        self.cache={key:value for key,value in self.cache.items() if key in {folder.name for folder in folders[:24]}}
        return rows
