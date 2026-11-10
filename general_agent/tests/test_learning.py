import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experience import Memory
from general_agent import learning

class LearningTests(unittest.TestCase):
    def test_memory_named_file_keeps_shared_families_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'experience.json').write_text(json.dumps({'facts':[{'id':'maze'}]}))
            self.assertEqual(Memory(root).facts[0]['id'],'maze')
            self.assertEqual(Memory(root,'web-runtime.experience.json').facts,[])

    def test_unknown_evidence_rejected_and_repeat_distillation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);common=root/'knowledge/common';common.mkdir(parents=True)
            evidence=[{'step':5}]
            parsed={'facts':[{'id':'valid','scope':'common','instruction':'Reobserve after a stale input.',
                             'evidence_steps':[5],'confidence':0.7},
                            {'id':'invalid','scope':'game','instruction':'Invented lesson.',
                             'evidence_steps':[999],'confidence':1}]}
            with patch.object(learning,'ROOT',root),patch.object(learning,'COMMON',common):
                first=learning.merge_reflection('game','run',parsed,evidence,root)
                second=learning.merge_reflection('game','run',parsed,evidence,root)
                self.assertEqual(len(first),1)
                self.assertEqual(second[0]['support'],1)
                self.assertEqual(len(second[0]['evidence']),1)
                text=learning.render_lessons(root/'knowledge/game','play')
                self.assertIn('Reobserve',text)
                self.assertNotIn('Invented',text)
                self.assertFalse((root/'knowledge/game/experience.json').exists())


    def test_game_only_reflection_leaves_shared_memory_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);common=root/'knowledge/common';common.mkdir(parents=True)
            shared=common/'web-runtime.experience.json'
            original=json.dumps({'facts':[{'id':'existing-common'}]})
            shared.write_text(original)
            parsed={'facts':[
                {'id':'game','scope':'game','instruction':'Use observed game resources.', 'evidence_steps':[2]},
                {'id':'common','scope':'common','instruction':'Verify the effect.', 'evidence_steps':[2]}]}
            with patch.object(learning,'ROOT',root),patch.object(learning,'COMMON',common):
                accepted=learning.merge_reflection('example','trial',parsed,[{'step':2}],root,scopes={'game'})
            self.assertEqual(len(accepted),1)
            self.assertEqual(accepted[0]['scope'],'game')
            self.assertEqual(shared.read_text(),original)
            self.assertTrue((root/'knowledge/example/experience.json').exists())


    def test_review_aligns_execution_with_its_own_action_and_not_previous_feedback(self):
        rows=[]
        for step,action,previous in [(0,'move',{}),(1,'upgrade',{'action':'move','phase_after':'levelup'}),
                                     (2,'',{'action':'upgrade','phase_after':'playing'})]:
            rows.append({'step':step,'observation':{'state':{'phase':'levelup' if step==1 else 'playing'},
                'actions':[{'id':action,'label':action}]},'decision':{'action_id':action,'by':'jev' if action else 'stop'},
                'last_execution':previous})
        evidence=learning.evidence_from_trace(rows,[{'action':'move','phase_after':'levelup'},
                                                   {'action':'upgrade','phase_after':'playing'}])
        self.assertEqual(evidence[0]['execution']['phase_after'],'levelup')
        self.assertEqual(evidence[1]['execution']['action'],'upgrade')
        self.assertEqual(evidence[1]['legal_actions'][0]['id'],'upgrade')
        self.assertEqual(evidence[1]['execution']['phase_after'],'playing')
        self.assertEqual(evidence[2]['execution'],{})
        shifted=learning.evidence_from_trace(rows)
        self.assertEqual(shifted[0]['execution']['action'],'move')
        self.assertEqual(shifted[1]['execution']['action'],'upgrade')


    def trajectory(self,count=80,transition=None):
        rows=[]
        for step in range(count):
            phase='second' if transition is not None and step>=transition else 'first'
            rows.append({'step':step,'observation':{'state':{'phase':phase,
                'runtime':{'visible_marker':20<=step<=60}},'actions':[{'id':'move','label':'move'}]},
                'decision':{'action_id':'move','by':'jev'},'last_execution':{}})
        return rows

    def test_review_includes_within_phase_play_instead_of_only_start_and_finish(self):
        evidence=learning.evidence_from_trace(self.trajectory())
        self.assertLessEqual(len(evidence),32)
        self.assertTrue(any(item['runtime']['visible_marker'] for item in evidence))
        self.assertIn(79,[item['step']for item in evidence])
        self.assertEqual([item['step']for item in evidence],sorted(item['step']for item in evidence))

    def test_review_pairs_observations_before_and_after_a_transition(self):
        evidence=learning.evidence_from_trace(self.trajectory(transition=40))
        steps={item['step']for item in evidence}
        self.assertTrue({38,39,40}.issubset(steps))
        self.assertLessEqual(len(evidence),32)
