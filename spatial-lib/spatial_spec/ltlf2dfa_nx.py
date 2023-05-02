import re

import networkx as nx
from ltlf2dfa.base import MonaProgram
from ltlf2dfa.ltlf2dfa import invoke_mona, createMonafile
from ltlf2dfa.parser.ltlf import LTLfParser


# Dump a value from a file based on a regex passed in.
def get_value(text, regex, value_type=str):
    pattern = re.compile(regex, re.MULTILINE)
    results = pattern.search(text)
    if results:
        return value_type(results.group(1))
    else:
        print("Could not find the value {}, in the text provided".format(regex))
        return value_type(0.0)


class LTLf2nxParser:

    def __init__(self):
        self.parser = LTLfParser()
        self.formula = None

    def parse_formula(self, formula_str):
        self.formula = self.parser(formula_str)

    def to_dot(self):
        if self.formula is None:
            print('<LTLf2ndParser.to_dot()> No formula parsed. Please parse a formula first using parse_formula(str)!')
            return None

        return self.formula.to_dfa()

    def to_mona_output(self):
        if self.formula is None:
            print('<LTLf2ndParser.to_mona_output()> No formula parsed. Please parse a formula first using parse_formula(str)!')
            return None

        mona_p_string = MonaProgram(self.formula).mona_program()
        createMonafile(mona_p_string)
        mona = invoke_mona()

        return mona

    def to_nxgraph(self, name='MONA_DFA'):
        if self.formula is None:
            print('<LTLf2ndParser.to_nxgraph()> No formula parsed. Please parse a formula first using parse_formula(str)!')
            return None

        mona_output = self.to_mona_output()
        g = nx.DiGraph()
        g.graph['name'] = name

        # if formula is unsatisfiable
        if "Formula is unsatisfiable" in mona_output:
            print('<LTLf2ndParser.to_nxgraph()> Formula is unsatisfiable, DFA not constructed!')
            return None

        if "DFA for formula with free variables:" not in mona_output:
            print('<LTLf2ndParser.to_nxgraph()> an unexpected error occurred. Is MONA properly installed?')
            return None

        # atomic propositions
        variables = get_value(mona_output, r'.*DFA for formula with free variables:[\s]*(.*?)\n.*', str)
        g.graph['ap'] = variables.lower().split()

        # accepting states
        accepting_states = get_value(mona_output, r".*Accepting states:[\s]*(.*?)\n.*", str)
        accepting_states = [
            str(x.strip()) for x in accepting_states.split() if len(x.strip()) > 0
        ]

        g.graph['acc'] = accepting_states

        # edges and states
        for line in mona_output.splitlines():
            if line.startswith("State "):

                orig_state = get_value(line, r".*State[\s]*(\d+):\s.*", str)
                dest_state = get_value(line, r".*state[\s]*(\d+)[\s]*.*", str)
                guard = get_value(line, r".*:[\s](.*?)[\s]->.*", str)
                if g.has_edge(orig_state, dest_state):
                    g.edges[orig_state, dest_state]['guard'].append(guard)
                else:
                    g.add_edge(orig_state, dest_state, guard=[guard])

        # remove the don't care state 0
        assert g.has_edge('0', '1')
        assert len(list(g.successors('0'))) == 1
        assert len(g.edges['0', '1']['guard']) == 1
        assert all(c == 'X' for c in g.edges['0', '1']['guard'][0])
        g.remove_node('0')

        # initial state
        g.graph['init'] = '1'

        return g
