import unittest
from general_agent.scene_memory import ObservedScenes

CONFIG={'scene_fields':['floor'],'snapshot_fields':['navigation','player'],
        'require_fields':['navigation'],'max_scenes':2}

class ObservedScenesTests(unittest.TestCase):
    def state(self,floor,targets):
        return {'floor':floor,'navigation':{'targets':targets},'player':{'attack':10}}

    def test_only_previously_seen_scenes_are_returned_and_copies_cannot_mutate_game(self):
        memory=ObservedScenes(CONFIG)
        state=self.state(1,[{'id':'gem','label':'visible gem'}])
        self.assertEqual(memory.remember(state,0),[])
        state['navigation']['targets'].clear()
        history=memory.remember(self.state(2,[]),1)
        self.assertEqual([item['scene']['floor'] for item in history],[1])
        self.assertEqual(history[0]['snapshot']['navigation']['targets'][0]['id'],'gem')
        history[0]['snapshot']['navigation']['targets'].clear()
        self.assertTrue(memory.remember(self.state(2,[]),2)[0]['snapshot']['navigation']['targets'])
        self.assertNotIn('action',history[0])

    def test_revisited_scene_updates_consumed_resources_and_non_scene_panel_preserves_map(self):
        memory=ObservedScenes(CONFIG)
        memory.remember(self.state(1,[{'id':'gem'}]),0)
        memory.remember(self.state(1,[]),1)
        memory.remember({'floor':1,'player':{'attack':14}},2)
        history=memory.remember(self.state(2,[]),3)
        self.assertEqual(history[0]['snapshot']['navigation']['targets'],[])
        self.assertEqual(history[0]['observed_at_step'],1)
        self.assertIn('stale',history[0]['status'])

    def test_scene_capacity_is_bounded_and_unconfigured_games_do_not_retain_maps(self):
        memory=ObservedScenes(CONFIG)
        for floor in [1,2,3]:history=memory.remember(self.state(floor,[]),floor)
        self.assertEqual([item['scene']['floor']for item in history],[2])
        self.assertEqual(ObservedScenes().remember(self.state(1,[]),0),[])


    def test_reset_starts_a_fresh_scene_memory_for_an_independent_episode(self):
        from unittest.mock import patch
        import tempfile
        from general_agent.adapters.web_runtime import WebRuntimeAdapter
        adapter=WebRuntimeAdapter('test',entry={'entry_path':'test.html'},config={'memory':CONFIG})
        adapter.scene_memory.remember(self.state(1,[]),0)
        with tempfile.TemporaryDirectory() as folder:
            adapter.artifacts=folder
            with patch('general_agent.adapters.web_runtime.BrowserHost'):
                adapter.reset()
        self.assertEqual(adapter.scene_memory.remember(self.state(2,[]),1),[])
