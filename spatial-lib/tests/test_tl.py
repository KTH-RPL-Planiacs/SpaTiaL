import unittest

import numpy as np

from spatial_spec.geometry import PolygonCollection, SpatialInterface, Circle, ObjectInTime
from spatial_spec.logic import Spatial


class RightMovingDummyObject(ObjectInTime):
    def __init__(self, x, y):
        super().__init__()
        self._name = 'blue'
        self.start_x = x
        self.start_y = y

    def getObject(self, time) -> 'SpatialInterface':
        return PolygonCollection({Circle(np.array([self.start_x + time, self.start_y]), 1)})

    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        pass


class LeftMovingDummyObject(ObjectInTime):
    def __init__(self, x, y):
        super().__init__()
        self._name = 'red'
        self.start_x = x
        self.start_y = y

    def getObject(self, time) -> 'SpatialInterface':
        return PolygonCollection({Circle(np.array([self.start_x - time, self.start_y]), 1)})

    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        pass


class UnmovingObject(ObjectInTime):
    def __init__(self, x, y):
        super().__init__()
        self._name = 'green'
        self.start_x = x
        self.start_y = y

    def getObject(self, time) -> 'SpatialInterface':
        return PolygonCollection({Circle(np.array([self.start_x, self.start_y]), 1)})

    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        pass


class TestTL(unittest.TestCase):

    def setUp(self):
        self.spatial = Spatial(quantitative=True)
        self.blue_t = RightMovingDummyObject(0, 0)
        self.green_t = UnmovingObject(50, 0)
        self.red_t = LeftMovingDummyObject(100, 0)

    def test_always(self):
        tree = self.spatial.parse('(G(!(blue touch green)))')

        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)

        x = range(0, 100, 1)
        y = []
        for i in x:
            y.append(self.spatial.interpret(tree, 0, i))

        for i in range(len(y) - 1):
            self.assertGreaterEqual(y[i], y[i + 1])  # the series should be monotone decreasing

        self.assertGreater(y[42], 0)
        self.assertLess(y[44], 0)
        self.assertLess(y[99], 0)

    def test_always_bounded(self):
        tree = self.spatial.parse('(G[0,20](!(blue touch green)))')

        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)

        x = range(0, 50, 1)
        y = []
        for i in x:
            y.append(self.spatial.interpret(tree, 0, i))

        for i in range(len(y) - 1):
            self.assertGreaterEqual(y[i], y[i + 1])  # the series should be monotone decreasing
            if i > 20:
                self.assertEqual(y[i], y[i + 1])  # the always is bounded

        self.assertGreater(y[-1], 0)  # accepting

    def test_eventually(self):
        tree = self.spatial.parse('(F(red touch green))')

        self.spatial.assign_variable('red', self.red_t)
        self.spatial.assign_variable('green', self.green_t)

        x = range(0, 100, 1)
        y = []
        for i in x:
            y.append(self.spatial.interpret(tree, 0, i))

        for i in range(len(y) - 1):
            self.assertLessEqual(y[i], y[i + 1])  # the series should be monotone increasing

        self.assertLess(y[42], 0)
        self.assertGreater(y[44], 0)  # circles are touching at timestep 43 because epsilon is default 5
        self.assertGreater(y[-1], 0)  # accepting

    def test_eventually_bounded(self):
        tree = self.spatial.parse('(F[0,20](red touch green))')

        self.spatial.assign_variable('red', self.red_t)
        self.spatial.assign_variable('green', self.green_t)

        x = range(0, 50, 1)
        y = []
        for i in x:
            y.append(self.spatial.interpret(tree, 0, i))

        for i in range(len(y) - 1):
            self.assertLessEqual(y[i], y[i + 1])  # the series should be monotone increasing
            if i > 20:
                self.assertEqual(y[i], y[i + 1])  # the eventually is bounded

        self.assertLess(y[-1], 0)  # accepting

    def test_until(self):
        tree = self.spatial.parse('((!(red touch green)) U (blue touch green))')

        green_t = UnmovingObject(30, 0)

        self.spatial.assign_variable('red', self.red_t)
        self.spatial.assign_variable('green', green_t)
        self.spatial.assign_variable('blue', self.blue_t)

        x = range(0, 100, 1)
        y = [self.spatial.interpret(tree, 0, i) for i in x]

        self.assertGreater(y[-1], 0)  # accepting

    def test_until_bounded(self):
        tree = self.spatial.parse('((!(red touch green)) U[0,20] (blue touch green))')

        green_t = UnmovingObject(30, 0)

        self.spatial.assign_variable('red', self.red_t)
        self.spatial.assign_variable('green', green_t)
        self.spatial.assign_variable('blue', self.blue_t)

        x = range(0, 100, 1)
        y = [self.spatial.interpret(tree, 0, i) for i in x]

        for i in range(21, len(y) - 1):
            self.assertEqual(y[i], y[i + 1])  # the until is bounded

        self.assertGreater(y[-1], 0)  # accepting

    def test_and(self):
        tree = self.spatial.parse('(G(!(blue touch green)))')
        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)
        always_result = self.spatial.interpret(tree, 0, 100)

        tree = self.spatial.parse('(F(red touch green))')
        self.spatial.assign_variable('red', self.red_t)
        self.spatial.assign_variable('green', self.green_t)
        eventually_result = self.spatial.interpret(tree, 0, 100)

        tree = self.spatial.parse('((G(!(blue touch green))) & (F(red touch green)))')
        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)
        self.spatial.assign_variable('red', self.red_t)
        combined_result = self.spatial.interpret(tree, 0, 100)

        self.assertIsNotNone(eventually_result)
        self.assertIsNotNone(always_result)
        self.assertIsNotNone(combined_result)
        self.assertEqual(combined_result, min(always_result, eventually_result))

    def test_or(self):
        tree = self.spatial.parse('(G(!(blue touch green)))')
        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)
        always_result = self.spatial.interpret(tree, 0, 100)

        tree = self.spatial.parse('(F(red touch green))')
        self.spatial.assign_variable('red', self.red_t)
        self.spatial.assign_variable('green', self.green_t)
        eventually_result = self.spatial.interpret(tree, 0, 100)

        tree = self.spatial.parse('((G(!(blue touch green))) | (F(red touch green)))')
        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)
        self.spatial.assign_variable('red', self.red_t)
        combined_result = self.spatial.interpret(tree, 0, 100)

        self.assertIsNotNone(eventually_result)
        self.assertIsNotNone(always_result)
        self.assertIsNotNone(combined_result)
        self.assertEqual(combined_result, max(always_result, eventually_result))

    def test_not(self):
        tree = self.spatial.parse('(G(!(blue touch green)))')
        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)
        always_result = self.spatial.interpret(tree, 0, 100)

        tree = self.spatial.parse('(!t(G(!(blue touch green))))')
        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)
        negated_result = self.spatial.interpret(tree, 0, 100)

        self.assertIsNotNone(always_result)
        self.assertIsNotNone(negated_result)
        self.assertEqual(negated_result, -always_result)

    def test_implies(self):
        tree = self.spatial.parse('(G(!(blue touch green)))')
        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)
        always_result = self.spatial.interpret(tree, 0, 100)

        tree = self.spatial.parse('(F(red touch green))')
        self.spatial.assign_variable('red', self.red_t)
        self.spatial.assign_variable('green', self.green_t)
        eventually_result = self.spatial.interpret(tree, 0, 100)

        tree = self.spatial.parse('((G(!(blue touch green))) ->t (F(red touch green)))')
        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)
        self.spatial.assign_variable('red', self.red_t)
        combined_result = self.spatial.interpret(tree, 0, 100)

        self.assertIsNotNone(eventually_result)
        self.assertIsNotNone(always_result)
        self.assertIsNotNone(combined_result)
        self.assertEqual(combined_result, max(-always_result, eventually_result))

    def test_next(self):
        tree = self.spatial.parse('(blue touch green)')
        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)
        result = self.spatial.interpret(tree, 1, 10)

        tree = self.spatial.parse('(X(blue touch green))')
        self.spatial.assign_variable('blue', self.blue_t)
        self.spatial.assign_variable('green', self.green_t)
        next_result = self.spatial.interpret(tree, 0, 10)

        self.assertIsNotNone(result)
        self.assertIsNotNone(next_result)
        self.assertEqual(result, next_result)


if __name__ == '__main__':
    unittest.main()
