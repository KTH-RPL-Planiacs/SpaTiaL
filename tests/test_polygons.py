import unittest

import matplotlib.pyplot as plt
import numpy as np

from spatial.geometry import Circle, PolygonCollection, Polygon


def to_bool(val: float) -> bool:
    return np.greater_equal(val, 0)


class TestPolygonSpatialRelations(unittest.TestCase):

    @staticmethod
    def approximate_polyline(line: np.ndarray, r: float = 10, d: float = 10) -> PolygonCollection:

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

        return PolygonCollection(circles)

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
        forbidden = Polygon(rectangle)  # self.approximate_polyline(rectangle,r=40)

        return PolygonCollection(red_polygons), PolygonCollection(green_polygons), \
               PolygonCollection(blue_polygons), PolygonCollection({forbidden})

    def setUp(self):
        red, green, blue, forbidden = self.setup1()
        self.red = red
        self.blue = blue
        self.green = green
        self.forbidden = forbidden

    def plot(self):
        self.red.draw()
        self.blue.draw()
        self.green.draw()
        plt.autoscale()
        plt.show()

    def test_left_of(self):
        self.assertTrue(to_bool(self.blue.left_of(self.red)))
        self.assertTrue(to_bool(self.green.left_of(self.red)))
        self.assertFalse(to_bool(self.red.left_of(self.green)))
        self.assertFalse(to_bool(self.red.left_of(self.blue)))

    def test_right_of(self):
        self.assertTrue(to_bool(self.red.right_of(self.blue)))
        self.assertTrue(to_bool(self.red.right_of(self.green)))
        self.assertFalse(to_bool(self.blue.right_of(self.red)))
        self.assertFalse(to_bool(self.green.right_of(self.red)))

    def test_above(self):
        self.assertTrue(to_bool(self.green.above(self.red)))
        self.assertTrue(to_bool(self.blue.above(self.red)))
        self.assertFalse(to_bool(self.red.above(self.green)))
        self.assertFalse(to_bool(self.red.above(self.blue)))

    def test_below(self):
        self.assertTrue(to_bool(self.red.below(self.blue)))
        self.assertTrue(to_bool(self.red.below(self.green)))
        self.assertFalse(to_bool(self.blue.below(self.red)))
        self.assertFalse(to_bool(self.green.below(self.red)))

    def test_close_to(self):
        self.assertFalse(to_bool(self.blue.close_to(self.red)))
        self.assertTrue(to_bool(self.blue.close_to(self.green)))
        self.assertFalse(to_bool(self.red.close_to(self.green)))

    def test_far_from(self):
        self.assertTrue(to_bool(self.blue.far_from(self.red)))
        self.assertFalse(to_bool(self.blue.far_from(self.green)))
        self.assertTrue(to_bool(self.red.far_from(self.green)))

    def test_touching(self):
        self.assertTrue(to_bool(self.red.touching(self.forbidden)))
        self.assertFalse(to_bool(self.red.touching(self.blue)))

    def test_overlapping(self):
        self.assertTrue(to_bool(self.blue.overlap(self.green)))
        self.assertTrue(to_bool(self.red.overlap(self.forbidden)))

    def test_proximity(self):
        r = 30
        c1 = PolygonCollection({Circle(np.array([0, 0]), r)})
        c2 = PolygonCollection({Circle(np.array([100, 0]), r)})
        self.assertTrue(
            to_bool(c1.proximity(c2, 100 - 2 * r)))  # strange number due to approximation of circle
        self.assertFalse(to_bool(c1.proximity(c2, 100 - 2 * r - 0.00001)))  # just a bit closer (0.0002)


if __name__ == '__main__':
    unittest.main()
