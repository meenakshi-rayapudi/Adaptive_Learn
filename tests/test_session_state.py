import datetime as dt
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ui.session_state as ss
from core.learner_service import SESSION_GAP_MINUTES


class FakeState(dict):
    """Stands in for st.session_state, which allows both state['x'] and state.x."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class TestSessionTimer(unittest.TestCase):
    def setUp(self):
        self.real_st = ss.st
        self.state = FakeState()
        ss.st = types.SimpleNamespace(session_state=self.state)

    def tearDown(self):
        ss.st = self.real_st

    def minutes_ago(self, n):
        return dt.datetime.now() - dt.timedelta(minutes=n)

    def test_first_visit_starts_a_session(self):
        ss.touch_session()
        self.assertIsNotNone(self.state["session_started_at"])
        self.assertFalse(ss.is_session_idle())

    def test_activity_inside_the_gap_keeps_the_same_session(self):
        started = self.minutes_ago(10)
        self.state.update(session_started_at=started, last_active_at=self.minutes_ago(3))
        ss.touch_session()
        self.assertEqual(self.state["session_started_at"], started)
        self.assertEqual(ss.session_minutes(), 10)

    def test_coming_back_after_the_gap_starts_a_new_session(self):
        long_ago = self.minutes_ago(SESSION_GAP_MINUTES + 20)
        self.state.update(session_started_at=long_ago, last_active_at=long_ago)
        self.assertTrue(ss.is_session_idle())
        ss.touch_session()
        self.assertEqual(ss.session_minutes(), 0)
        self.assertFalse(ss.is_session_idle())

    def test_no_session_means_zero_minutes(self):
        self.assertEqual(ss.session_minutes(), 0)

    def test_switching_student_restarts_the_timer_and_updates_the_agent(self):
        agent = types.SimpleNamespace(student_id="old")
        self.state.update(session_started_at=self.minutes_ago(9), last_active_at=self.minutes_ago(1),
                          agent_executor=agent)
        ss.set_active_student("new")
        self.assertEqual(self.state["student_id"], "new")
        self.assertEqual(agent.student_id, "new")
        self.assertEqual(ss.session_minutes(), 0)

    def test_switching_student_without_an_agent_is_fine(self):
        ss.set_active_student("new")
        self.assertEqual(self.state["student_id"], "new")


if __name__ == "__main__":
    unittest.main()
