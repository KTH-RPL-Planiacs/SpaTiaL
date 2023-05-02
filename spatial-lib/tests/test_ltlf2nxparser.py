import unittest

from spatial_spec.ltlf2dfa_nx import LTLf2nxParser


class TestLTLf2nxParser(unittest.TestCase):

    def test_noformula(self):
        parser = LTLf2nxParser()
        self.assertIsNone(parser.to_mona_output())
        self.assertIsNone(parser.to_nxgraph())
        self.assertIsNone(parser.to_dot())

    def test_unsatisfiable(self):
        parser = LTLf2nxParser()
        parser.parse_formula("F (a & !a)")
        self.assertTrue("Formula is unsatisfiable" in parser.to_mona_output())
        self.assertIsNone(parser.to_nxgraph())

    def test_satisfiable(self):
        parser = LTLf2nxParser()
        parser.parse_formula("G (a -> F b)")
        automaton = parser.to_nxgraph()
        self.assertIsNotNone(automaton)
        self.assertEqual(len(automaton.nodes), 2)
        self.assertEqual(len(automaton.edges), 4)
        self.assertEqual(len(automaton.graph['acc']), 1)
        self.assertEqual(len(automaton.graph['init']), 1)
