import unittest

from spatial.automaton_planning import AutomatonPlanner
from spatial.logic import Spatial


class TestAutomatonPlanning(unittest.TestCase):

    def setUp(self) -> None:
        spatial = Spatial()
        self.planner = AutomatonPlanner()

        ex = "( (G(!(blue touch red))) & (F(blue touch green)) )"
        ex_tree = spatial.parse(ex)
        self.planner.tree_to_dfa(ex_tree)

    def test_automaton_creation(self):
        self.planner.reset_state()

        self.assertEqual(self.planner.temporal_formula, "(G(a)) & (F(b))")
        self.assertEqual(len(self.planner.dfa.nodes), 4)
        self.assertEqual(len(self.planner.dfa.edges), 9)

    def test_trace_simulation(self):
        trace_ap = ['a', 'b']
        ex_trace_acc = ['10', '10', '10', '11']
        ex_trace_rej = ['10', '00', '10', '01']

        self.planner.reset_state()
        self.assertTrue(self.planner.simulate_trace(ex_trace_acc, trace_ap))
        self.planner.reset_state()
        self.assertFalse(self.planner.simulate_trace(ex_trace_rej, trace_ap))

    def test_planning(self):
        trace_ap = ['a', 'b']  # uppercase required
        self.planner.reset_state()
        self.planner.dfa_step('10', trace_ap)  # initial observation
        ex_target, ex_constraint = self.planner.plan_step()
        if self.planner.get_dfa_ap() == ['a', 'b']:
            self.assertEqual(ex_target, {'11'})
            self.assertEqual(ex_constraint, {'0X'})
        else:
            self.assertEqual(ex_target, {'11'})
            self.assertEqual(ex_constraint, {'X0'})

        # staying inside constraint for 5 times, nothing should change
        for i in range(5):
            self.planner.dfa_step('10', ['a', 'b'])
            ex_target, ex_constraint = self.planner.plan_step()
            self.assertFalse(self.planner.currently_accepting())

        # satisfying target, automaton should accept
        self.planner.dfa_step('11', ['a', 'b'])
        self.assertTrue(self.planner.currently_accepting())
        ex_target, ex_constraint = self.planner.plan_step()
        self.assertIsNone(ex_constraint)  # constraint and target is the same if we are in a goal state
