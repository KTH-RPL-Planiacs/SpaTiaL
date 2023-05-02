import unittest

import numpy as np

from spatial_spec.geometry import Circle, Polygon, PolygonCollection, StaticObject, DynamicObject
from spatial_spec.logic import Spatial


class TestSpatialCircles(unittest.TestCase):

    @staticmethod
    def approximate_polyline(line: np.ndarray, r: float = 10, d: float = 10) -> StaticObject:

        circles = set()

        for i in range(len(line)):
            diff = line[(i + 1) % len(line)] - line[i]
            seg_len = np.linalg.norm(diff)
            steps = np.arange(d, seg_len - d, d) / seg_len
            # dx=x2-x1 and dy=y2-y1, then the normals are (-dy, dx) and (dy, -dx).
            n = np.array([-diff[1], diff[0]])
            n = (n / np.linalg.norm(n)) * r
            for s in steps:
                circles.add(Circle(line[i] + s * diff + n, r))

        return StaticObject(PolygonCollection(circles))

    def setup1(self):

        # circle sets
        red_circles = set()
        red_circles.add(Circle(np.array([350, 250]), 50))
        red_circles.add(Circle(np.array([300, 100]), 60))
        green_circles = set()
        green_circles.add(Circle(np.array([100, 350]), 55))
        blue_circles = set()
        blue_circles.add(Circle(np.array([100, 300]), 50))

        # constraints
        rectangle = np.array([[400, 0], [600, 0], [600, 500], [400, 500]])
        forbidden = self.approximate_polyline(rectangle, r=40)

        return StaticObject(PolygonCollection(red_circles)), StaticObject(PolygonCollection(blue_circles)), \
               StaticObject(PolygonCollection(green_circles)), forbidden

    def setUp(self):
        red, blue, green, forbidden = self.setup1()
        self.forbidden = forbidden
        self.spatial = Spatial()
        self.spatial.assign_variable('red', red)
        self.spatial.assign_variable('green', green)
        self.spatial.assign_variable('blue', blue)
        self.spatial.assign_variable('f', forbidden)

    def test_left_of(self):
        formula = "(blue leftof red) and (green leftof red) and (not ((red leftof green) and (red leftof blue)))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_right_of(self):
        formula = "(red rightof blue) and (red rightof green) and (not ((blue rightof red) and (green rightof red)))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_above(self):
        formula = "(green above red) and (blue above red) and (not ((red above green) and (red above blue)))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_below(self):
        formula = "(red below blue) and (red below green) and (not ((blue below red) and (green below red)))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_close_to(self):
        formula = "(not (blue closeto red)) and (blue closeto green) and (not (red closeto green))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))
        formula = "(blue closer green than red)"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_far_from(self):
        formula = "(blue farfrom red) and (not (blue farfrom green)) and (red farfrom green)"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_touching(self):
        formula = "(red touch f) and (not (red touch blue))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_overlapping(self):
        formula = "(blue ovlp green) and ((red ovlp f))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_proximity(self):
        r = 30
        c1 = StaticObject(PolygonCollection({Circle(np.array([0, 0]), r)}))
        c2 = StaticObject(PolygonCollection({Circle(np.array([100, 0]), r)}))

        self.spatial.assign_variable('c1', c1)
        self.spatial.assign_variable('c2', c2)
        self.spatial.assign_variable('r1', 100 - 2 * r)
        self.spatial.assign_variable('r2', 100 - 2 * r - 0.1)

        formula = "(c1 dist c2 <= r1) and (not (c1 dist c2 <= r2))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_quantitative(self):
        spatial = Spatial(quantitative=True)

        r = 30
        c1 = StaticObject(PolygonCollection({Circle(np.array([0, 0]), r)}))
        c2 = StaticObject(PolygonCollection({Circle(np.array([100, 0]), r)}))

        spatial.assign_variable('c1', c1)
        spatial.assign_variable('c2', c2)
        spatial.assign_variable('r1', 100 - 2 * r)

        formula = "(c1 ovlp c2)"
        # circle = list(c2.getObject(0).shapes())[0]
        for i in range(12):
            c2 = StaticObject(PolygonCollection({Circle(np.array([100 - i * 10, 0]), r)}))
            spatial.assign_variable('c2', c2)
            # print(spatial.parse_and_interpret(formula))
            if 100 - i * 10 - 2 * r >= 0:
                self.assertGreaterEqual(0, spatial.parse_and_interpret(formula))
            else:
                self.assertGreater(spatial.parse_and_interpret(formula), 0)


class TestSpatialPolygons(unittest.TestCase):

    @staticmethod
    def polygon_from_circle(c: Circle) -> np.ndarray:

        vertices = np.array(
            [c.c + c.r * np.array([np.cos(theta), np.sin(theta)]) for theta in np.linspace(0, np.pi * 2, 20)])
        return Polygon(vertices, c.color)

    @staticmethod
    def setup1():

        # polygon sets
        red_polygons = set()
        red_polygons.add(Circle(np.array([350, 250]), 50))
        red_polygons.add(Circle(np.array([300, 100]), 60))
        green_polygons = set()
        green_polygons.add(Circle(np.array([100, 350]), 55))
        blue_polygons = set()
        blue_polygons.add(Circle(np.array([100, 300]), 50))

        # constraints
        rectangle = np.array([[400, 0], [600, 0], [600, 500], [400, 500]])
        forbidden = Polygon(rectangle)

        return StaticObject(PolygonCollection(red_polygons)), StaticObject(PolygonCollection(green_polygons)), \
               StaticObject(PolygonCollection(blue_polygons)), StaticObject(PolygonCollection({forbidden}))

    def setUp(self):
        red, green, blue, forbidden = self.setup1()
        self.spatial = Spatial()
        self.spatial.assign_variable('red', red)
        self.spatial.assign_variable('green', green)
        self.spatial.assign_variable('blue', blue)
        self.spatial.assign_variable('f', forbidden)

    def test_left_of(self):
        formula = "(blue leftof red) and (green leftof red) and (not ((red leftof green) and (red leftof blue)))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_right_of(self):
        formula = "(red rightof blue) and (red rightof green) and (not ((blue rightof red) and (green rightof red)))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_above(self):
        formula = "(green above red) and (blue above red) and (not ((red above green) and (red above blue)))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_below(self):
        formula = "(red below blue) and (red below green) and (not ((blue below red) and (green below red)))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_close_to(self):
        formula = "(not (blue closeto red)) and (blue closeto green) and (not (red closeto green))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))
        formula = "(blue closer green than red)"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_far_from(self):
        formula = "(blue farfrom red) and (not (blue farfrom green)) and (red farfrom green)"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_touching(self):
        formula = "(red touch f) and (not (red touch blue))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_overlapping(self):
        formula = "(blue ovlp green) and ((red ovlp f))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_proximity(self):
        r = 30
        c1 = StaticObject(PolygonCollection({Circle(np.array([0, 0]), r)}))
        c2 = StaticObject(PolygonCollection({Circle(np.array([100, 0]), r)}))

        self.spatial.assign_variable('c1', c1)
        self.spatial.assign_variable('c2', c2)
        self.spatial.assign_variable('r1', 100 - 2 * r)
        self.spatial.assign_variable('r2', 100 - 2 * r - 0.0001)

        formula = "(c1 dist c2 <= r1) and (not (c1 dist c2 <= r2))"
        self.assertTrue(self.spatial.parse_and_interpret(formula))

    def test_quantitative(self):
        spatial = Spatial(quantitative=True)

        r = 30
        c1 = StaticObject(PolygonCollection({Circle(np.array([0, 0]), r)}))
        # start at 110 due to first translation of -10 at i=0 in for loop
        c2 = StaticObject(PolygonCollection({Circle(np.array([110, 0]), r)}))

        spatial.assign_variable('c1', c1)
        spatial.assign_variable('c2', c2)
        spatial.assign_variable('r1', 100 - 2 * r)

        formula = "(c1 ovlp c2)"
        polygon = list(c2.getObject(0).polygons)[0]
        for i in range(12):
            polygon.translate(np.array([-10, 0]))  # translate by 10 each step

            # c1.draw()
            # c2.draw()
            # plt.autoscale()
            # plt.show()

            if 100 - i * 10 - 2 * r + 0.5 >= 0:  # +0.5 due to circle approximation with polygon
                self.assertGreaterEqual(0, spatial.parse_and_interpret(formula))
            else:
                self.assertGreater(spatial.parse_and_interpret(formula), 0)


class TestRelativeSpatial(unittest.TestCase):

    @staticmethod
    def setup1():
        center = np.array([0, 0])
        red = DynamicObject()
        ground_truth = [True] * len(range(30 - 1))
        ground_truth[9] = False  # circle +1 is moving
        ground_truth[19] = False  # circle +1 is moving
        for i in range(30):
            if 0 <= i < 10:
                red.addObject(PolygonCollection({Circle(center + [0, 0], 3)}), i)
            if 10 <= i < 20:
                red.addObject(PolygonCollection({Circle(center + [1, 0], 3)}), i)
            if 20 <= i:
                red.addObject(PolygonCollection({Circle(center + [2, 0], 3)}), i)
        return red, ground_truth

    @staticmethod
    def setup2():
        center = np.array([0, 0])
        inner_set = DynamicObject()
        inner_set.addObject(PolygonCollection({Circle(center + [-1.5, 0], 1), Circle(center + [1.5, 0], 1)}), 0)
        inner_set.addObject(PolygonCollection({Circle(center + [-1.5, 0], 1), Circle(center + [4.5, 0], 1)}), 1)
        inner_set.addObject(PolygonCollection({Circle(center + [-1.5, 0], 1), Circle(center + [4.5, 0], 1)}), 2)

        outer = DynamicObject()
        outer.addObject(PolygonCollection({Circle(center, 5)}), 0)
        outer.addObject(PolygonCollection({Circle(center, 5)}), 1)
        return inner_set, outer

    @staticmethod
    def to_bool(val):
        return True if val >= 0 else False

    def setUp(self):
        red, ground_truth = self.setup1()
        self.gt = ground_truth
        self.red = red
        self.spatial = Spatial(quantitative=False)
        self.spatial.assign_variable('red', red)

        self.inner, self.outer = self.setup2()
        self.spatial.assign_variable('inner', self.inner)
        self.spatial.assign_variable('outer', self.outer)

    def test_within_single(self):
        # go through dynamics list for single object
        for i in range(30 - 1):
            enclosed = self.red.getObject(i).enclosed_in(self.red.getObject(i + 1))
            self.assertEqual(self.to_bool(enclosed), self.gt[i])

    def test_within_multi(self):
        # small circles of inner lie within large circle of outer => True
        self.assertTrue(self.to_bool(self.inner.getObject(0).enclosed_in(self.outer.getObject(0))))

        # rightmost small circle of inner moved out of large circle of outer => False
        self.assertFalse(self.to_bool(self.inner.getObject(1).enclosed_in(self.outer.getObject(0))))

        # check each instances of inner
        self.assertTrue(self.to_bool(self.inner.getObject(0).enclosed_in(self.inner.getObject(0))))
        self.assertFalse(self.to_bool(self.inner.getObject(0).enclosed_in(self.inner.getObject(1))))

    def test_within_multi_spatial_interface(self):
        formula = "(G(inner enclosedin outer))"
        self.assertTrue(self.spatial.interpret(self.spatial.parse(formula), 0, 0))
        self.assertFalse(self.spatial.interpret(self.spatial.parse(formula), 0, 1))

    def test_relative_within(self):
        formula = "(F(inner enclosedin inner[-1]))"
        tree = self.spatial.parse(formula)

        # evaluate from 0 to 0 -> returns NAN
        self.assertTrue(np.isnan(self.spatial.interpret(tree, 0, 0)))

        # evaluate from 0 to 2 -> returns NAN
        self.assertTrue(np.isnan(self.spatial.interpret(tree, 0, 2)))

        # evaluate from 1 to 1 -> returns False
        self.assertFalse(self.spatial.interpret(tree, 1, 1))

        # evaluate from 1 to 2 -> returns True
        self.assertTrue(self.spatial.interpret(tree, 1, 2))


class TestLogicMethods(unittest.TestCase):

    def test_min_time(self):
        formula1 = "(red[-1] leftof green[-4])"
        formula2 = "(red[-1] leftof green)"
        formula3 = "(red leftof green)"

        spatial = Spatial()
        tree = spatial.parse(formula1)
        self.assertEqual(spatial.min_time_of_formula(tree), 4.)

        tree = spatial.parse(formula2)
        self.assertEqual(spatial.min_time_of_formula(tree), 1.)

        tree = spatial.parse(formula3)
        self.assertEqual(spatial.min_time_of_formula(tree), 0)


if __name__ == '__main__':
    unittest.main()
