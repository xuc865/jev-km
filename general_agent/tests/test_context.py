import unittest
from copy import deepcopy
from general_agent.context import prompt_state


class ContextProjectionTests(unittest.TestCase):
    def test_runtime_keeps_public_facts_and_does_not_mutate_trace_state(self):
        state = {'phase':'play','instructions':'public rules','last_execution':{'ok':True},
                 'fast_state':{'phase':'play'},'runtime':{'phase':'play',
                    'screen':'all visible text','flags':{'selecting_target':True},
                    'navigation':{'grid':['..#'], 'targets':[{'id':'a'},{'id':'b'}]},
                    'unknown_public_fact':42,'_dom':{'text':'all visible text','candidates':[]}}}
        original=deepcopy(state)
        view=prompt_state(state)
        self.assertNotIn('fast_state',view)
        self.assertNotIn('_dom',view['runtime'])
        for key,value in state['runtime'].items():
            if key!='_dom':self.assertEqual(view['runtime'][key],value)
        self.assertEqual(view['instructions'],state['instructions'])
        self.assertEqual(view['last_execution'],state['last_execution'])
        view['runtime']['navigation']['targets'].clear()
        self.assertEqual(state,original)

    def test_non_runtime_state_is_preserved(self):
        state={'fast_state':{'unique_fact':1},'_dom':{'custom_public_fact':2}}
        self.assertEqual(prompt_state(state),state)


if __name__=='__main__':unittest.main()
