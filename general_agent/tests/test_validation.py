import json
from pathlib import Path
import tempfile
import unittest
from general_agent.validation import assess, episode_metrics, build

class ValidationTests(unittest.TestCase):
    def fixture(self,root,trial='acceptance_fresh_start',outer=True):
        (root/'browser').mkdir()
        (root/'summary.json').write_text(json.dumps({'game':'fixture','run_id':'test','result':{'success':True,'stopped':False}}))
        (root/'protocol.json').write_text(json.dumps({'trial_type':trial,'assistant_gameplay_actions':0,'record_video':True}))
        (root/'browser/host.json').write_text(json.dumps({'recording':True}))
        (root/'browser/full-run.webm').write_bytes(bytes.fromhex('1a45dfa3')+b'fixture-only'*100)
        (root/'browser/final.png').write_bytes(b'fixture-only')
        (root/'trace.jsonl').write_text(json.dumps({'decision':{'model':'WeMM-JEV'},
            'outer_calls':[{'model':'DeepSeek-test'}] if outer else []})+'\n')

    def test_step_victory_is_not_accepted_as_realtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            protocol=json.loads((root/'protocol.json').read_text())
            protocol['runtime_config']={'time':{'mode':'step'}}
            (root/'protocol.json').write_text(json.dumps(protocol))
            self.assertTrue(assess(root)['accepted'])
            report=assess(root,required_time_mode='realtime')
            self.assertFalse(report['accepted'])
            self.assertTrue(report['base_accepted'])
            self.assertEqual(report['timing']['validation_class'],'stepped_feasibility')
            protocol['runtime_config']['time']['mode']='realtime'
            (root/'protocol.json').write_text(json.dumps(protocol))
            self.assertTrue(assess(root,required_time_mode='realtime')['accepted'])

    def test_missing_clock_cannot_be_inferred_as_realtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            report=assess(root,required_time_mode='realtime')
            self.assertFalse(report['accepted'])
            self.assertEqual(report['timing']['mode'],'unknown')

    def test_diagnostic_save_continuation_is_not_acceptance_even_after_victory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root,trial='diagnostic_save_continuation')
            report=assess(root)
            self.assertTrue(report['success']);self.assertFalse(report['accepted'])
            self.assertFalse(report['checks']['fresh_start'])

    def test_both_models_are_required_for_video_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root,outer=False)
            report=assess(root)
            self.assertFalse(report['accepted'])
            self.assertFalse(report['checks']['deepseek_called'])


    def test_suite_picks_latest_finished_success_instead_of_sorting_random_run_ids(self):
        import os
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runs=root/'runs';runs.mkdir()
            older=runs/'zzz-old';newer=runs/'aaa-new'
            older.mkdir();newer.mkdir()
            for index,trial in enumerate([older,newer]):
                self.fixture(trial)
                summary=json.loads((trial/'summary.json').read_text())
                summary['run_id']=trial.name
                (trial/'summary.json').write_text(json.dumps(summary))
                os.utime(trial/'summary.json',(100+index,100+index))
            suite=root/'suite.json'
            suite.write_text(json.dumps({'name':'test','games':[{'id':'fixture'}]}))
            report=build(suite,runs)
            self.assertEqual(report['games'][0]['accepted_run']['run_id'],'aaa-new')



class EpisodeMetricsTests(unittest.TestCase):
    def test_counts_final_execution_from_summary_without_double_counting_trace_ack(self):
        execution={'ok':True,'changed':False}
        rows=[{'decision':{'by':'jev','model':'WeMM-JEV','ms':100},'last_execution':{}},
              {'decision':{'by':'escalate','model':'WeMM-JEV','ms':200},
               'last_execution':execution,'outer_calls':[{'model':'DeepSeek-test','ms':600}]}]
        report=episode_metrics({'executions':[execution,{'ok':True,'changed':True}]},rows)
        self.assertEqual(report['jev_calls'],2)
        self.assertEqual(report['deepseek_calls'],1)
        self.assertEqual(report['jev_latency']['median_ms'],150)
        self.assertEqual(report['jev_latency']['p95_ms'],200)
        self.assertEqual(report['executions']['count'],2)
        self.assertEqual(report['executions']['acknowledged'],2)
        self.assertEqual(report['executions']['state_changed'],1)

    def test_missing_latency_is_unknown_instead_of_zero(self):
        report=episode_metrics({},[{'decision':{'by':'stop'}}])
        self.assertIsNone(report['jev_latency']['median_ms'])
        self.assertIsNone(report['deepseek_latency']['p95_ms'])
        self.assertEqual(report['executions']['count'],0)
