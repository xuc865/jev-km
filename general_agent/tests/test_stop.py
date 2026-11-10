import unittest

from general_agent.policy import ExecutorPolicy
from general_agent.runner import run_episode
from general_agent.types import Action, Observation

from .fake_models import ScriptedClient, executor_reply


class StopTests(unittest.TestCase):
    def test_stop_during_inference_does_not_execute_the_planned_input(self):
        stop = [False]
        actions = []
        closed = []

        class Adapter:
            def reset(self):
                pass

            def observe(self):
                return Observation(actions=[Action("play")])

            def act(self, action):
                actions.append(action.id)

            def close(self):
                closed.append(True)

        class StopAfterFirstCall(ScriptedClient):
            """Flips the stop flag while the model call is in flight.

            This is the case the test exists for: the action was already chosen
            when the operator hit stop, so the loop must not send it.
            """

            def complete(self, *args, **kwargs):
                stop[0] = True
                return super().complete(*args, **kwargs)

        client = StopAfterFirstCall(executor=[executor_reply("play")])
        result = run_episode(
            Adapter(),
            ExecutorPolicy(client=client, model="fake"),
            "win",
            should_stop=lambda: stop[0],
        )
        self.assertTrue(result.stopped)
        self.assertEqual(actions, [])
        self.assertTrue(closed)


if __name__ == "__main__":
    unittest.main()
