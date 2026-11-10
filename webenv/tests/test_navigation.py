import unittest
from webenv.navigation import reachable_targets

KEYS = {'up':'ArrowUp','down':'ArrowDown','left':'ArrowLeft','right':'ArrowRight'}

class NavigationTests(unittest.TestCase):
    def test_paths_do_not_consume_other_targets_on_the_way(self):
        navigation={'position':[1,1], 'walkable':['.'], 'grid':['#######','#.k.m.#','#######'],
                    'targets':[{'id':'key','x':2,'y':1,'label':'key'},
                               {'id':'monster','x':4,'y':1,'label':'monster'}]}
        targets=reachable_targets(navigation,KEYS)
        self.assertEqual([t['id'] for t in targets], ['key'])
        self.assertEqual(targets[0]['keys'], ['ArrowRight'])

    def test_no_reward_ranking_and_no_unavailable_interactions(self):
        navigation={'position':[2,2], 'walkable':['.'], 'grid':['#####','#.a.#','#...#','#.b.#','#####'],
                    'targets':[{'id':'lower','x':2,'y':3,'label':'coin'},
                               {'id':'upper','x':2,'y':1,'label':'gem'}]}
        self.assertEqual([t['id'] for t in reachable_targets(navigation,KEYS)], ['lower','upper'])
        navigation['targets'][0]['available']=False
        self.assertEqual([t['id'] for t in reachable_targets(navigation,KEYS)], ['upper'])

    def test_empty_detour_is_compiled_to_normal_keys(self):
        navigation={'position':[1,1], 'walkable':['.'], 'grid':['#####','#.#x#','#...#','#####'],
                    'targets':[{'id':'target','x':3,'y':1,'label':'target'}]}
        target=reachable_targets(navigation,KEYS)[0]
        self.assertEqual(target['keys'],['ArrowDown','ArrowRight','ArrowRight','ArrowUp'])
        self.assertEqual(target['positions'][-1],[3,1])


class OccupiedTargetContracts(unittest.TestCase):
    def test_current_target_is_reentered_via_an_empty_neighbor(self):
        keys={'up':'ArrowUp','down':'ArrowDown','left':'ArrowLeft','right':'ArrowRight'}
        navigation={'position':[1,1],'grid':['###','#>.','###'],'walkable':['.'],
                    'targets':[{'id':'stairs','x':1,'y':1,'label':'stairs'}]}
        result=reachable_targets(navigation,keys)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['keys'],['ArrowRight','ArrowLeft'])
        self.assertEqual(result[0]['positions'],[[2,1],[1,1]])
        self.assertEqual(result[0]['steps'],2)


class NavigationEligibilityTests(unittest.TestCase):
    def test_visible_resource_behind_an_interaction_is_not_marked_executable(self):
        from copy import deepcopy
        from webenv.navigation import describe_reachability
        keys={'up':'U','down':'D','left':'L','right':'R'}
        navigation={'position':[1,1],'grid':['#####','#.ab#','#####'],'walkable':['.'],
                    'targets':[{'id':'a','x':2,'y':1,'available':True},
                               {'id':'b','x':3,'y':1,'available':True}]}
        original=deepcopy(navigation)
        described=describe_reachability(navigation,keys)
        self.assertTrue(described['targets'][0]['executable_now'])
        self.assertTrue(described['targets'][1]['resource_eligible'])
        self.assertFalse(described['targets'][1]['executable_now'])
        self.assertEqual(navigation,original)
        navigation['grid'][1]='#..b#'
        navigation['targets']=navigation['targets'][1:]
        self.assertTrue(describe_reachability(navigation,keys)['targets'][0]['executable_now'])

    def test_reachability_annotations_have_stable_input_signatures(self):
        from types import SimpleNamespace
        from general_agent.adapters.web_runtime import WebRuntimeAdapter,signature
        navigation={'position':[0,0],'grid':['.a'],'targets':[{'id':'a','x':1,'y':0,'available':True,'label':'visible target'}]}
        original={'navigation':navigation}
        adapter=WebRuntimeAdapter('test',config={'navigation':{'keys':{'up':'U','down':'D','left':'L','right':'R'}}})
        adapter.host=SimpleNamespace(snapshot=lambda:original)
        first=adapter._read();second=adapter._read()
        self.assertEqual(signature(first),signature(second))
        self.assertEqual(adapter._actions(first)[0].params['signature'],signature(second))
        self.assertNotIn('executable_now',navigation['targets'][0])
