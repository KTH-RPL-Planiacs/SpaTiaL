import os
import pickle
import warnings
from typing import Dict

import numpy as np
from lark import Lark, Transformer, Tree, v_args
from lark.tree import pydot__tree_to_graph
from lark.visitors import Interpreter

from spatial_spec.geometry import SpatialInterface, ObjectInTime


@v_args(inline=True)  # Affects the signatures of the methods
class SpatRelInterpreter(Transformer):
    """
    Interpreter for spatial relations. Delegates parsed tree to corresponding operations.
    """

    # from operator import neg, and_ as b_and, or_ as b_or
    # from operator import and_, or_, not_
    number = float

    def __init__(self):
        """
        Initializes the interpreter
        """
        super().__init__()
        self.vars: Dict[str, ObjectInTime] = {}
        self.number_vars: Dict[str, float] = {}
        self._global_time = 0

    def set_global_time(self, time: int):
        assert time >= 0, "<Interpreter>: global time must be non-negative! Got: {}".format(time)
        self._global_time = time

    @staticmethod
    def spatial(a):
        """
        Maps quantitative semantics to bool domain
        Args:
            a: The float value

        Returns: True if val>=0 and False otherwise

        """
        return a

    @staticmethod
    def and_(a, b) -> float:
        """
        Computes the quantitative semantics of AND operator
        Args:
            a: predicate left of operator
            b: predicate right of operator

        Returns: min(a,b)

        """
        return np.min([a, b])

    @staticmethod
    def or_(a, b) -> float:
        """
        Computes the quantitative semantics of OR operator
        Args:
            a: predicate left of operator
            b: predicate right of operator

        Returns: max(a,b)

        """
        return np.max([a, b])

    @staticmethod
    def xor_(a, b) -> float:
        """
        Computes the quantitative semantics of XOR operator
        Args:
            a: predicate left of operator
            b: predicate right of operator

        Returns: max(a,b)

        """
        # a XOR b = (a & !b) | (!a & b)
        a_notb = np.min([a, -b])
        nota_b = np.min([-a, b])
        return np.max([a_notb, nota_b])

    @staticmethod
    def implies_(a, b) -> float:
        """
        Computes the quantitative semantics of IMPLIES operator
        Args:
            a: predicate left of operator
            b: predicate right of operator

        Returns: max(a,b)

        """
        # a -> b = !a | b
        return np.max([-a, b])

    @staticmethod
    def not_(a) -> float:
        """
        Computes the quantitative semantics of not operator
        Args:
            a: predicate to negate

        Returns: -a

        """
        return -a

    @property
    def vars(self) -> dict:
        """
        Dictionary of stored (name, variable) pairs
        Returns: dictionary

        """
        return self._vars

    @vars.setter
    def vars(self, vars: dict):
        """
        Sets the dictionary of stored (name, variable) pairs
        Args:
            vars: Dictionary

        """
        if len(vars) > 1:
            elements = list(vars.values())
            assert all([isinstance(k, type(elements[0])) for k in
                        elements]), '<SpatialInterpreter>: only one type of obstacle currently supported!'
        self._vars = vars

    @property
    def number_vars(self) -> dict:
        """
        Returns the dictionary of stored (name, numerical variable) pairs
        Returns: dictionary

        """
        return self._number_vars

    @number_vars.setter
    def number_vars(self, number_vars: dict):
        """
        Sets the dictionary of stored (name, numerical variable) pairs
        Args:
            number_vars: dictionary

        """
        assert len(number_vars) == 0 or all([isinstance(k, (float, int)) for k in
                                             [number_vars.values()]]), '<SpatialInterpreter>: only numbers supported!'
        self._number_vars = number_vars

    def assign_var(self, name: str, value: ObjectInTime):
        """
        Assigns a new variable to the interpreter
        Args:
            name: Name of the variable
            value: Value of the variables (spatial interface object or int/float)

        """

        assert isinstance(name, str), '<Logic>: name must be of type string! Got {}'.format(name)

        if isinstance(value, (int, float)):
            self.number_vars[name.lower()] = value
        else:

            assert isinstance(value, ObjectInTime), '<Logic>: value must be of type ' \
                                                    'ObjectInTime! Got {}'.format(
                value)

            self.vars[name.lower()] = value
        return value

    def var(self, name: str) -> SpatialInterface:
        """
        Returns the variable corresponding to the provided name
        Args:
            name: Name of the variable

        Returns: spatial interface object of name

        """

        # query latest time step
        return self.var_at(name, 0)

    def var_at(self, name: str, rel_time: int) -> SpatialInterface:
        """
        Returns the variable corresponding to the provided name and time step
        Args:
            name: Name of the variable
            rel_time: relative time identifier

        Returns: spatial interface object of name

        """

        assert rel_time >= 0., ''

        try:
            obj: ObjectInTime = self.vars[name.lower()]
        except KeyError:
            raise Exception("Variable not found: %s" % name)

        try:
            return obj.getObject(self._global_time - rel_time)
        except Exception:
            raise Exception("Time step index t={} not found".format(self._global_time - rel_time))

    @staticmethod
    def enlarge(o: SpatialInterface, radius: float):
        return o.enlarge(radius)

    def numeric_var(self, name):
        """
        Returns the numeric variable of a specified name
        Args:
            name: The specified name

        Returns: int/float object corresponding to specified name

        """
        try:
            return self.number_vars[name.lower()]
        except KeyError:
            raise Exception("Variable not found: %s" % name)

    @staticmethod
    def left_of(left: SpatialInterface, right: SpatialInterface):
        return left.left_of(right)

    @staticmethod
    def right_of(left: SpatialInterface, right: SpatialInterface):
        return left.right_of(right)

    @staticmethod
    def below_of(left: SpatialInterface, right: SpatialInterface):
        return left.below(right)

    @staticmethod
    def above_of(left: SpatialInterface, right: SpatialInterface):
        return left.above(right)

    @staticmethod
    def overlap(left: SpatialInterface, right: SpatialInterface):
        return left.overlap(right)

    @staticmethod
    def touching(left: SpatialInterface, right: SpatialInterface):
        return left.touching(right)

    @staticmethod
    def far_from(left: SpatialInterface, right: SpatialInterface):
        return left.far_from(right)

    @staticmethod
    def close_to(left: SpatialInterface, right: SpatialInterface):
        return left.close_to(right)

    @staticmethod
    def enclosed_in(left: SpatialInterface, right: SpatialInterface):
        return left.enclosed_in(right)

    @staticmethod
    def comparison(left: SpatialInterface, right: SpatialInterface):
        return [left, right]

    @staticmethod
    def moved(left: SpatialInterface, right: SpatialInterface):
        val = left.enclosed_in(right.enlarge(25))
        return val if np.isclose(val, 0) else -val

    @staticmethod
    def operator(value):
        """
        Maps operators (<=,>=,=) to numpy functions
        Args:
            value: The operator to map

        Returns: Numpy functions object

        """
        # "<=" | ">=" | "=="
        if value == "<=":
            return np.less_equal
        if value == ">=":
            return np.greater_equal
        if value == "==":
            return np.equal

    @staticmethod
    def closer_to(left: SpatialInterface, right: list):
        return left.closer_to_than(right[0], right[1])

    @staticmethod
    def distance(left: SpatialInterface, right: SpatialInterface, fun, eps):
        return left.distance_compare(right, eps, fun)


# custom function wrapper for the SpatialInterpreter.visit() function
def _vargs_tree_time(f, data, children, meta, lower, upper):
    return f(Tree(data, children, meta), lower, upper)


@v_args(wrapper=_vargs_tree_time)
class SpatialInterpreter(Interpreter):
    """
    Interpreter object for the temporal parts of Spatial. Delegates parsed tree to corresponding operations.
    """

    def __init__(self, spatial, quantitative: bool = False):
        self._quantitative = quantitative
        self._spatial_interpreter = spatial
        self._spatial_dict = {}
        self.vars = {}

    @property
    def quantitative(self) -> bool:
        """
        Bool whether this interpreter returns boolean or quantitative values

        Returns: True if quantitative

        """
        return self._quantitative

    @quantitative.setter
    def quantitative(self, quantitative: bool):
        """
        Bool whether this interpreter returns boolean or quantitative values
        Args:
            quantitative: Set a new bool

        """
        self._quantitative = quantitative

    @property
    def vars(self) -> dict:
        """
        Dictionary of stored (name, variable) pairs
        Returns: dictionary

        """
        return self._vars

    @vars.setter
    def vars(self, vars: dict):
        """
        Sets the dictionary of stored (name, variable) pairs
        Args:
            vars: Dictionary

        """
        if len(vars) > 1:
            elements = list(vars.values())
            assert all([isinstance(k, ObjectInTime) for k in
                        elements]), '<SpatialInterpreter>: only ObjectInTime currently supported!'
        self._vars = vars

    def assign_var(self, name: str, value: ObjectInTime):
        """
        Assigns a new variable to the interpreter
        Args:
            name: Name of the variable
            value: Value of the variables (ObjectInTime interface object or int/float)

        """
        if isinstance(value, (int, float)):
            self._spatial_interpreter.assign_var(name, value)
        else:

            assert isinstance(value,
                              ObjectInTime), '<SpatialInterpreter>: value must be of type ' \
                                             'ObjectInTime or int/float! Got {}'.format(
                value)
            assert isinstance(name, str), '<SpatialInterpreter>: name must be of type string! Got {}'.format(name)

            self.vars[name.lower()] = value
        return value

    def var(self, name: str) -> ObjectInTime:
        """
        Returns the variable corresponding to the provided name
        Args:
            name: Name of the variable

        Returns: spatial interface object of name

        """
        try:
            return self.vars[name.lower()]
        except KeyError:
            raise Exception("Variable not found: %s" % name)

    # translates the relative bounds of bounded temporal operators into the absolute time
    @staticmethod
    def relative_to_absolute_bounds(rel_lower, rel_upper, lower, upper):
        assert rel_lower >= 0 and rel_upper >= 0, \
            '<SpatialInterpreter>: negative bounds in bounded temporal operators not allowed!'
        assert rel_lower <= rel_upper, \
            '<SpatialInterpreter>: relative lower bound is higher than relative upper bound'

        abs_lower = lower + rel_lower
        abs_upper = min(upper, lower + rel_upper)

        return abs_lower, abs_upper

    # hacky override of original function. provides necessary extra parameters for custom function wrapper
    def visit(self, tree, lower, upper):
        f = getattr(self, tree.data)
        wrapper = getattr(f, 'visit_wrapper', None)
        if wrapper is not None:
            return f.visit_wrapper(f, tree.data, tree.children, tree.meta, lower, upper)
        else:
            return f(tree)

    def temporal(self, tree, lower, upper):
        # temporal has only one child
        return self.visit(tree.children[0], lower, upper)

    def and_(self, tree, lower, upper):
        # and_ has two children
        left = self.visit(tree.children[0], lower, upper)
        right = self.visit(tree.children[1], lower, upper)
        return np.nanmin([left, right])

    def or_(self, tree, lower, upper):
        # or_ has two children
        left = self.visit(tree.children[0], lower, upper)
        right = self.visit(tree.children[1], lower, upper)
        return np.nanmax([left, right])

    def xor_(self, tree, lower, upper):
        # xor_ has two children
        left = self.visit(tree.children[0], lower, upper)
        right = self.visit(tree.children[1], lower, upper)
        # a XOR b = (a & !b) | (!a & b)
        a_notb = np.nanmin([left, -right])
        nota_b = np.nanmin([-left, right])
        return np.nanmax([a_notb, nota_b])

    def implies_(self, tree, lower, upper):
        # implies_ has two children
        left = self.visit(tree.children[0], lower, upper)
        right = self.visit(tree.children[1], lower, upper)
        return np.nanmax([-left, right])  # works because a -> b == !a v b

    def not_(self, tree, lower, upper):
        # not_ has a single child
        return -self.visit(tree.children[0], lower, upper)

    def eventually(self, tree, lower, upper):
        results = []
        for i in range(lower, upper + 1):
            results.append(self.visit(tree.children[0], i, upper))
            # speedup in case the interpreter is run in boolean mode
            if not self.quantitative and results[-1] >= 0:
                return 1.
        return np.nanmax(results)

    def eventually_bounded(self, tree, lower, upper):
        bound = tree.children[0]
        rel_bound_l = int(bound.children[0].children[0])
        rel_bound_u = int(bound.children[1].children[0])

        abs_bound_l, abs_bound_u = self.relative_to_absolute_bounds(rel_bound_l, rel_bound_u, lower, upper)
        # this happens when the relative lower bound references a point in time later than upper
        if abs_bound_l > abs_bound_u:
            return np.nan

        # create a 'fake' eventually tree and interpret it over new bounds
        eventually_tree = Tree('eventually', [tree.children[1]])
        return self.eventually(eventually_tree, abs_bound_l, abs_bound_u)

    def always(self, tree, lower, upper):
        # always has only one child
        results = []
        for i in range(lower, upper + 1):
            results.append(self.visit(tree.children[0], i, upper))
            # speedup in case the interpreter is run in boolean mode
            if not self.quantitative and results[-1] < 0:
                return -1.
        return np.nanmin(results)

    def always_bounded(self, tree, lower, upper):
        bound = tree.children[0]
        rel_bound_l = int(bound.children[0].children[0])
        rel_bound_u = int(bound.children[1].children[0])

        abs_bound_l, abs_bound_u = self.relative_to_absolute_bounds(rel_bound_l, rel_bound_u, lower, upper)
        # this happens when the relative lower bound references a point in time later than upper
        if abs_bound_l > abs_bound_u:
            return np.nan

        # create a 'fake' always tree and interpret it over new bounds
        always_tree = Tree('always', [tree.children[1]])
        return self.always(always_tree, abs_bound_l, abs_bound_u)

    def next(self, tree, lower, upper):
        if lower + 1 > upper:
            return np.nan

        # next always has only one child
        return self.visit(tree.children[0], lower + 1, upper)

    until_storage = dict()

    def until(self, tree, lower, upper):
        # final result
        result = -np.inf

        # store results of current tree evaluation in lookup table
        element = hash((tree.children[0], tree.children[1]))
        if element not in self.until_storage:
            self.until_storage[element] = {}
        # get dictionary of previous calls
        # stores calls of self.visit(tree.children[0], j, k). key is (i, k), value is result
        v2_dict = self.until_storage[element]

        for k in range(lower, upper + 1):
            v1 = self.visit(tree.children[1], k, upper)
            # this whole section is simply
            # v2 = min(self.visit(tree.children[0], j, k) for j in range(lower, k+1))
            v2 = np.inf
            for j in range(lower, k + 1):
                interval = (j, k)
                if interval not in v2_dict:
                    val = self.visit(tree.children[0], j, k)
                    v2_dict[interval] = val
                    if val < v2:
                        v2 = val
                else:
                    val = v2_dict[interval]
                    if val < v2:
                        v2 = val
            val = np.nanmin([v1, v2])
            if val > result:
                result = val
        return result

    def until_bounded(self, tree, lower, upper):
        bound = tree.children[1]
        rel_bound_l = int(bound.children[0].children[0])
        rel_bound_u = int(bound.children[1].children[0])

        abs_bound_l, abs_bound_u = self.relative_to_absolute_bounds(rel_bound_l, rel_bound_u, lower, upper)
        # this happens when the relative lower bound references a point in time later than upper
        if abs_bound_l > abs_bound_u:
            return np.nan

        # create a 'fake' until tree and interpret it over new bounds
        until_tree = Tree('until', [tree.children[0], tree.children[2]])
        return self.always(until_tree, abs_bound_l, abs_bound_u)

    def spatial(self, tree, lower, upper):
        # check if this spatial formula has already been evaluated for this time point
        element = hash((tree, lower))  # compute hash once
        if element not in self._spatial_dict:
            # set global time for evaluation of formula
            self._spatial_interpreter.set_global_time(lower)
            val = self._spatial_interpreter.transform(tree)
            self._spatial_dict[element] = val
        return self._spatial_dict[element]


class Spatial(object):
    """
    Spatial parser (+ interpreter)
    """

    def __init__(self, quantitative: bool = False):
        """
        Initializes the Spatial object
        Args:
            quantitative: True if quantitative semantics are desired
        """
        grammar = os.path.dirname(__file__) + "/spatial.lark"
        self._parser = Lark.open(grammar, parser='lalr')
        self._spatial_interpreter = SpatRelInterpreter()
        self._tl_interpreter = SpatialInterpreter(self._spatial_interpreter, quantitative=quantitative)
        self.quantitative = quantitative

    @property
    def quantitative(self) -> bool:
        return self._quantitative

    @quantitative.setter
    def quantitative(self, quantitative: bool):
        self._quantitative = quantitative
        self._tl_interpreter.quantitative = quantitative

    def reset_spatial_dict(self):
        """
        Resets the spatial interpreter call history
        """
        self._tl_interpreter._spatial_dict = {}

    def parse(self, formula: str) -> Tree:
        """
        Parses a given formula to a tree object
        Args:
            formula: Formula to parse

        Returns: Tree object of parsed formula

        """
        try:
            self.reset_spatial_dict()  # every time you parse a new formula, reset the spatial dict
            return self._parser.parse(formula)
        except Exception as e:
            print(e)
            return None

    def interpret(self, formula: Tree, lower=0, upper=0):
        """
        Interprets a given tree
        Args:
            formula: The tree of the formula
            lower: lower time bound for semantics
            upper: upper time bound for semantics

        Returns: Value of interpreted formula

        """

        # check if relative time has been used
        rel_time = self.min_time_of_formula(formula)
        # return NAN if start time does not allow to evaluate formula
        if rel_time > lower:
            warnings.warn(f"<Interpreter/interpret>: Cannot evalute formula from time step {lower} "
                          f"since the formalue is valid from {rel_time}!")
            return np.NAN

        try:
            val = self._tl_interpreter.visit(formula, lower, upper)
            if self.quantitative:
                return val
            else:
                return val >= 0
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def svg_from_tree(formula: Tree, filename, rankdir="LR", **kwargs):
        """
        Saves tree object to a svg vector graphics file
        Args:
            formula: The tree of the formula
            filename: The filename
            rankdir: params for pydot
            **kwargs: params for pydot

        """
        graph = pydot__tree_to_graph(formula, rankdir, **kwargs)
        graph.write_svg(filename)

    @staticmethod
    def png_from_tree(formula: Tree, filename, rankdir="LR", **kwargs):
        """
        Saves tree object to a png graphics file
        Args:
            formula: The tree of the formula
            filename: The filename
            rankdir: params for pydot
            **kwargs: params for pydot

        """
        graph = pydot__tree_to_graph(formula, rankdir, **kwargs)
        graph.write_png(filename)

    @staticmethod
    def determine_variables(formula: Tree):
        """
        Determines all variables within a formula (given as a tree)
        Args:
            formula: The formula to check

        Returns: Set of all variable names

        """
        iter = formula.find_data('var')
        vars = set()
        for v in iter:
            vars.add(v.children[0].title())
        return vars

    def check_variables(self, formula: Tree) -> bool:
        """
        Checks if the interpreter stores all variables required to interpret a given formula
        Args:
            formula: The formula to check as a tree

        Returns: True if interpreter stores all necessary variables and False otherwise

        """
        vars = self.determine_variables(formula)
        for v in vars:
            if v.lower() not in self._tl_interpreter.vars.keys():
                return False
        return True

    @staticmethod
    def min_time_of_formula(formula: Tree) -> int:
        """
        If the formula contains relative time references (variable plus time reference), then the formula can only
        be evaluated when the minimum time has been reached in the interpreter. This function returns the minimum
        required time to evaluate the formula.
        Args:
            formula: The formula to check

        Returns: The minimum time rquired to evaluate the formula

        """

        iter = formula.find_data('var_at')  # tag used for variables with time reference
        vars = [0]
        for v in iter:
            time = float(v.children[1].children[0].title())
            assert time >= 0
            vars.append(time)
        return np.max(vars)

    def update_variables(self, vars: dict):
        """
        Update the set of variables in the interpreter
        Args:
            vars: The new set of variables

        """
        self._tl_interpreter.vars = vars

    def assign_variable(self, name, value):
        """
        Assigns a variable to the interpreter
        Args:
            name: Name of the variable
            value: Value of the variable

        """

        # if isinstance(value, ObjectInTime):
        #    self._tl_interpreter.assign_var(name, value)
        # elif isinstance(value, SpatialInterface):
        #    self._tl_interpreter.assign_var(name, StaticObject(value))
        # else:
        self._spatial_interpreter.assign_var(name, value)

    def parse_and_interpret(self, formula: str):
        """
        Parses and interprets a given formula
        Args:
            formula: Formula as string

        Returns:

        """
        self.reset_spatial_dict()
        return self.interpret(self.parse(formula))

    def save_to_file(self, file: str):
        """
        Saves the state of the interpreter to the files system
        Args:
            file: The filename

        """
        try:
            pickle.dump(self._spatial_interpreter, open(file, 'wb'))
        except Exception as e:
            print(e)

    def from_file(self, file: str):
        """
        Restores an interpreter from the file system
        Args:
            file: The filename

        """
        try:
            self._spatial_interpreter = pickle.load(open(file, 'rb'))
        except Exception as e:
            print(e)

    @staticmethod
    def write_formulas_to_file(file: str, formulas: list):
        """
        Writes a list of formulas to the file system
        Args:
            file: The filename
            formulas: The list of formulas to store

        """
        try:
            pickle.dump(formulas, open(file, 'wb'))
        except Exception as e:
            print(e)

    @staticmethod
    def load_formulas_from_file(file: str) -> list:
        """
        Loads a list of formulas from the file system
        Args:
            file: The filename

        Returns: list of formulas as strings

        """
        try:
            return pickle.load(open(file, 'rb'))
        except Exception as e:
            print(e)


if __name__ == "__main__":
    print("WOW")
