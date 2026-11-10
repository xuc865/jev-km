import json
from pathlib import Path
import tempfile
import unittest

from general_agent.monitor import EpisodeMonitor, trace_tail


class IndependentMonitorTests(unittest.TestCase):
    def write_run(self, root):
        folder=root/'test-run';folder.mkdir()
        (folder/'protocol.json').write_text(json.dumps({'game':'fixture','trial_type':'acceptance_fresh_start'}))
        row={'step':8,'elapsed':90,'decision':{'action_id':'move','by':'jev','model':'fixture'},
             'observation':{'state':{'runtime':{'phase':'play','score':1}},'actions':[]}}
        (folder/'trace.jsonl').write_text(json.dumps(row)+'\n')
        return folder

    def test_partial_trace_write_preserves_last_complete_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=self.write_run(Path(temporary))
            with (folder/'trace.jsonl').open('a') as handle:handle.write('{"step":9,"decision":')
            rows=trace_tail(folder/'trace.jsonl')
            self.assertEqual([row['step'] for row in rows],[8])
            monitor=EpisodeMonitor(Path(temporary))
            item=monitor.snapshots(now=(folder/'trace.jsonl').stat().st_mtime+4)[0]
            self.assertEqual(item['trace_age_seconds'],4)
            self.assertFalse(item['running'])
            self.assertTrue(item['external_snapshot'])

    def test_terminal_result_overrides_recent_activity_and_stopped_is_not_win(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=self.write_run(Path(temporary));monitor=EpisodeMonitor(Path(temporary))
            monitor.snapshots()
            (folder/'summary.json').write_text(json.dumps({'result':{'success':True,'stopped':True,'steps':9}}))
            item=monitor.snapshots()[0]
            self.assertTrue(item['finalized']);self.assertFalse(item['success'])
            self.assertEqual(item['status'],'已结束 · 未通关')

    def test_stale_unfinalized_and_diagnostic_runs_are_not_presented_as_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=self.write_run(Path(temporary));monitor=EpisodeMonitor(Path(temporary))
            self.assertEqual(monitor.snapshots(now=(folder/'trace.jsonl').stat().st_mtime+601),[])
            (folder/'protocol.json').write_text(json.dumps({'trial_type':'diagnostic'}))
            self.assertEqual(monitor.snapshots(),[])
