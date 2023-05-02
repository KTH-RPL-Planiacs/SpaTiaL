from copy import deepcopy

import networkx as nx

from spatial_spec.logic import Spatial
from spatial_spec.ltlf2dfa_nx import LTLf2nxParser


def number_to_base(n, b):
    """
    translates a decimal number to a different base in a list of digits
    Args:
        n: the decimal number
        b: the new base
    Returns:
        A list of digits in the new base
    """
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def sog_fits_to_guard(guard, sog, guard_ap, sog_ap):
    """
    checks if a guard fits to a set of guards, including undecided bits denoted as X
    Args:
        guard: A single guard. Example 'X01'
        sog: A set of guards Example {'001','101'}
        guard_ap: atomic propositions of the guard
        sog_ap: atomic propositions of the set of guards
    Returns:
        A subset of the set of guards matching to the single guard
    """

    guards = deepcopy(sog)

    # check if this guard fits to one of the guards
    for i, g_value in enumerate(guard):
        if g_value == 'X':
            continue
        if guard_ap[i] in sog_ap:
            j = sog_ap.index(guard_ap[i])
            wrong_guards = []
            for guard in guards:
                if guard[j] != 'X' and guard[j] != g_value:
                    # if a synth guard is not matching to the current config, mark for removal
                    wrong_guards.append(guard)
            # remove marked guards
            for wg in wrong_guards:
                guards.remove(wg)

    return guards


def flip_guard_bit(guard, bit, skip_ones=False):
    # construct hypothetical test_guard
    test_guard = list(guard)
    if guard[bit] == '1':
        if skip_ones:
            test_guard[bit] = '1'
        else:
            test_guard[bit] = '0'
    elif guard[bit] == '0':
        test_guard[bit] = '1'
    elif guard[bit] == 'X':
        test_guard[bit] = 'X'
    test_guard = ''.join(test_guard)
    return test_guard


def replace_guard_bit(guard, bit, lit):
    new_guard = list(guard)
    new_guard[bit] = lit
    new_guard = ''.join(new_guard)
    return new_guard


def resolve_all_x(guard):
    return_set = set()
    for i, o in enumerate(guard):
        if o == 'X':
            return_set = return_set.union(resolve_all_x(replace_guard_bit(guard, i, '0')))
            return_set = return_set.union(resolve_all_x(replace_guard_bit(guard, i, '1')))

    if len(return_set) == 0:
        return_set.add(guard)
    return return_set


def compare_obs(test_obs, existing_obs):
    if len(test_obs) != len(existing_obs):
        return False

    for i in range(len(test_obs)):
        if test_obs[i] == 'X' and existing_obs[i] == 'X':
            continue
        if test_obs[i] != existing_obs[i]:
            return False

    return True


def reduce_set_of_guards(sog):
    # first, resolve all generalizations
    new_sog = set()
    for g in sog:
        new_sog = new_sog.union(resolve_all_x(g))
    changed = True
    # blow up the set of guards with all possible generalizations
    while changed:
        blown_up_sog = set(new_sog)
        for guard in new_sog:  # for each guard
            for i, o in enumerate(guard):  # for each bit in the guard
                # construct hypothetical test_guard
                test_guard = flip_guard_bit(guard, i)

                # check if the test_guard and the guard are different (they are not if 'X' got "flipped")
                if test_guard == guard:
                    continue

                # if the hypothetical guard is not also in the new sog, we cannot reduce
                test_guard_in_new_sog = False
                for g in new_sog:
                    if compare_obs(test_guard, g):
                        test_guard_in_new_sog = True
                if not test_guard_in_new_sog:
                    continue

                # create a reduced guard
                reduced_guard = replace_guard_bit(guard, i, 'X')
                # add it to the reduced sog
                blown_up_sog.add(reduced_guard)

        if new_sog == blown_up_sog:
            changed = False

        new_sog = blown_up_sog

    # sog is all blown up with all possible generalizations
    unnecessary_guards = set()
    for guard in new_sog:
        for other_guard in new_sog:
            # skip yourself
            if guard == other_guard:
                continue

            # check if the guard subsumes the other_guard
            subsumes = True
            for j in range(len(guard)):
                if guard[j] == 'X':
                    continue
                if other_guard[j] == 'X':
                    subsumes = False
                if not guard[j] == other_guard[j]:
                    subsumes = False
            if subsumes:
                unnecessary_guards.add(other_guard)

    return new_sog.difference(unnecessary_guards)


class AutomatonPlanner:
    """
    Enables online automaton-based planning of the temporal aspects.
    """

    def __init__(self):
        """
        Initializes the AutomatonPlanner
        """
        self.ltlf_parser = LTLf2nxParser()

        self.spatial_dict = {}
        self.next_free_variable_id = 0

        self.temporal_formula = None
        self.dfa = None
        self.current_state = None

    def currently_accepting(self):
        """
        Returns True if the current state is accepting
        """
        return self.current_state in self.dfa.graph['acc']

    def reset_state(self):
        """
        Resets the current state to the initial state
        """
        self.current_state = self.dfa.graph['init']

    def simulate_trace(self, trace, ap):
        """
        Simulates a run on the automaton, given some inputs and updates the state accordingly

        Args:
            trace: The input trace
            ap: the input atomic propositions
        """
        self.reset_state()
        for symbol in trace:
            self.dfa_step(symbol, ap)
        return self.current_state in self.dfa.graph['acc']

    def dfa_step(self, symbol, ap):
        """
        Executes a single step of the automaton

        Args:
            symbol: The input symbol
            ap: the input atomic propositions
        """
        assert isinstance(symbol, str), 'Symbol is of wrong type, should be str! %s' % type(symbol)
        for succ in self.dfa.successors(self.current_state):
            # directly applying the first edge that fits works because the dfa is deterministic!
            if sog_fits_to_guard(symbol, self.dfa.edges[self.current_state, succ]['guard'], ap, self.dfa.graph['ap']):
                self.current_state = succ
                break

    def plan_step(self):
        """
        Returns the desired transition and a selfloop to maintain current state
        based on the shortest path to any accepting state.

        Returns:
            target_next_edge, self_loop_constraint, edge: If no path exists, target_next_edge is None.
            If target_next_edge is a self-loop (holding accepting state), self_loop_constraint is None.
            edge refers to the targeted edge in the DFA and can be used as a reference for pruning.
        """
        if self.current_state in self.dfa.graph['acc']:
            # if we are in an accepting state, hold it!
            return self.dfa.edges[self.current_state, self.current_state]['guard'], None, None

        # find shortest path to all accepting states
        targets = {}
        path_exists = False
        for acc_node in self.dfa.graph['acc']:
            try:
                targets[acc_node] = nx.shortest_path(self.dfa, source=self.current_state, target=acc_node)
                path_exists = True
            except nx.exception.NetworkXNoPath:
                targets[acc_node] = float('inf')

        if not path_exists:
            return None, None, None

        # find the closest reachable accepting state
        target = min(targets.items(), key=lambda x: len(x[1]))
        target_path = target[1]
        target_next_guards = self.dfa.edges[self.current_state, target_path[1]]['guard']
        target_constraint_guards = []
        for succ in self.dfa.successors(self.current_state):
            if succ != target_path[1] and succ != self.current_state:
                target_constraint_guards.extend(self.dfa.edges[self.current_state, succ]['guard'])
        return reduce_set_of_guards(target_next_guards), reduce_set_of_guards(target_constraint_guards), (self.current_state, target_path[1])

    def tree_to_dfa(self, tree):
        """
        Converts a SpaTiaL tree to a DFA.
        Args:
            tree: input tree parsed by spatial lark parser
        """
        self.temporal_formula = self.extract_temporal(tree)
        self.ltlf_parser.parse_formula(self.temporal_formula)
        self.dfa = self.ltlf_parser.to_nxgraph()

    def get_next_variable_string(self):
        """
        Creates a new, unique variable name.

        Returns: the variable string
        """
        id_base_26 = number_to_base(self.next_free_variable_id, 26)
        letters = [chr(n + ord('a')) for n in id_base_26]
        self.next_free_variable_id += 1
        return ''.join(letters)

    def extract_temporal(self, tree):
        """
        Takes a parsed tree and extracts an LTLf formula, creating variables for each "spatial"-type subtree.
        Keeps dictionaries for mappings between subtrees and variable names.

        Args:
            tree: input tree parsed by spatial lark parser

        Returns:
            LTLf formula represented as a string.
        """
        f = ''

        if tree.data == 'temporal':
            f += self.extract_temporal(tree.children[0])
        elif tree.data == 'always':
            f += 'G(' + self.extract_temporal(tree.children[0]) + ')'
        elif tree.data == 'eventually':
            f += 'F(' + self.extract_temporal(tree.children[0]) + ')'
        elif tree.data == 'until':
            f += '(' + self.extract_temporal(tree.children[0]) + ') U (' + self.extract_temporal(tree.children[1]) + ')'
        elif tree.data == 'and_':
            f += '(' + self.extract_temporal(tree.children[0]) + ') & (' + self.extract_temporal(tree.children[1]) + ')'
        elif tree.data == 'or_':
            f += '(' + self.extract_temporal(tree.children[0]) + ') | (' + self.extract_temporal(tree.children[1]) + ')'
        elif tree.data == 'not_':
            f += '!(' + self.extract_temporal(tree.children[0]) + ')'
        elif tree.data == 'implies_':
            f += '(' + self.extract_temporal(tree.children[0]) + ') -> (' + self.extract_temporal(tree.children[1]) + ')'
        elif tree.data == 'spatial':
            element = hash(tree)  # compute hash once
            # map hash to variable name and tree
            if element not in self.spatial_dict.keys():
                variable_string = self.get_next_variable_string()
                self.spatial_dict[element] = {
                    'variable': variable_string,
                    'tree': tree
                }
            f += self.spatial_dict[element]['variable']
        else:
            assert False, 'Operator %s not implemented for automaton-based planning!' % tree.data

        return f

    def get_variable_to_tree_dict(self):
        """
        Returns:
            A mapping between temporal variable name and associated spatial tree.
        """
        return_dict = {}

        for val in self.spatial_dict.values():
            return_dict[val['variable']] = val['tree']

        return return_dict

    def get_dfa_ap(self):
        """
        Returns:
             The atomic propositions of the DFA.
        """
        return self.dfa.graph['ap']


if __name__ == '__main__':
    spatial = Spatial()
    planner = AutomatonPlanner()

    # parse the formula
    ex = "( (G(!(blue touch red))) & (F(blue touch green)) )"
    ex_tree = spatial.parse(ex)  # build the usual tree
    planner.tree_to_dfa(ex_tree)  # transform the tree into an automaton

    # print the corresponding LTLf formula
    print(planner.temporal_formula)

    # this dictionary contains a variable name to spatial tree mapping
    print(planner.get_variable_to_tree_dict())

    # you have to define in which order you pass variable assignments
    trace_ap = ['a', 'b']

    # example traces (sequences of observations)
    ex_trace_acc = ['10', '10', '10', '11']
    ex_trace_rej = ['10', '00', '10', '01']

    # the automaton can evaluate such traces
    print('Trace 1 is', planner.simulate_trace(ex_trace_acc, trace_ap))
    print('Trace 2 is', planner.simulate_trace(ex_trace_rej, trace_ap))

    # INTERACTIVE PLANNING EXAMPLE
    # resets the automaton current state to the initial state
    planner.reset_state()

    # before you ask anything from the automaton, provide a initial observation
    planner.dfa_step('10', trace_ap)

    # planning loop
    print('')
    print('INTERACTIVE EXAMPLE')
    print('')
    print('enter observations in order', planner.get_dfa_ap(), 'or enter \'exit\' to exit')
    while True:

        target_set, constraint_set, _ = planner.plan_step()
        print('Currently accepting:', planner.currently_accepting())
        if target_set:
            print('Next step: Reach', target_set, planner.get_dfa_ap(),
                  ', never satisfy', constraint_set, planner.get_dfa_ap())
        else:
            print('No path exists!')
        obs = input('Next Observation:')
        if obs == 'exit':
            break
        planner.dfa_step(obs, planner.get_dfa_ap())
