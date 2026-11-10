"""Explicit clock semantics shared by host, recordings and acceptance reports."""
from __future__ import annotations
import math


def timing_contract(config: dict | None = None) -> dict:
    settings=(config or {}).get('time') or {}
    mode=settings.get('mode','step')
    if mode not in {'step','realtime'}:
        raise ValueError(f'Unknown time mode: {mode!r}; expected step or realtime')
    quantum=float(settings.get('dt_ms',1000/60))
    if not math.isfinite(quantum) or quantum<=0:
        raise ValueError('time.dt_ms must be a finite positive action quantum')
    paused=mode=='step'
    return {'mode':mode,'game_paused_during_inference':paused,
            'clock_source':'virtual-step' if paused else 'native-browser',
            'simulation_dt_ms':quantum if paused else None,
            'action_quantum_ms':quantum,
            'validation_class':'stepped_feasibility' if paused else 'realtime',
            'label':'STEP · 等待模型时游戏暂停' if paused else 'REALTIME · 推理时游戏持续运行'}
