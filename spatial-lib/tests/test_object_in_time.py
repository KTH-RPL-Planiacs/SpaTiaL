import unittest

import numpy as np

from spatial_spec.geometry import StaticObject, DynamicObject, ObjectCollection, PolygonCollection, Polygon, Circle


class TestStaticObject(unittest.TestCase):

    def test_add(self):
        c = Circle(np.array([0, 0]), 3)
        static = StaticObject(c)
        self.assertEqual(c, static.getObject(0))
        self.assertEqual(c, static.getObject(100))
        self.assertEqual(c, static.getObject(-100))


class TestDynamicObject(unittest.TestCase):

    def test_add(self):
        c0 = Circle(np.array([0, 0]), 3)
        c1 = Circle(np.array([0, 0]), 3)
        c2 = Circle(np.array([0, 0]), 3)
        dynamic = DynamicObject()

        # add first time step t=0
        dynamic.addObject(c0, 0)

        # check if object can be retrieved
        self.assertEqual(c0, dynamic.getObject(0))  # using absolute
        self.assertEqual(c0, dynamic.getObjectByIndex(0))  # using relative = 0
        self.assertEqual(c0, dynamic.getObjectByIndex(-1))  # using relative = 0

        # check if assertions work
        with self.assertRaises(AssertionError):
            dynamic.getObject(1)
        with self.assertRaises(AssertionError):
            dynamic.getObjectByIndex(-2)
        with self.assertRaises(AssertionError):
            dynamic.getObjectByIndex(1)

        # add more objects
        dynamic.addObject(c1, 1)
        dynamic.addObject(c2, 2)

        # check if objects can be retrieved
        self.assertEqual(c1, dynamic.getObject(1))  # using absolute
        self.assertEqual(c1, dynamic.getObjectByIndex(1))  # using relative = 0
        self.assertEqual(c1, dynamic.getObjectByIndex(-2))  # using relative = 0

        self.assertEqual(c2, dynamic.getObject(2))  # using absolute
        self.assertEqual(c2, dynamic.getObjectByIndex(2))  # using relative = 0
        self.assertEqual(c2, dynamic.getObjectByIndex(-1))  # using relative = 0

        # check if assertion works when skipping time step
        with self.assertRaises(AssertionError):
            dynamic.addObject(c0, 5)

        # check if getObject assertions work
        with self.assertRaises(AssertionError):
            dynamic.getObject(10)
        with self.assertRaises(AssertionError):
            dynamic.getObjectByIndex(-5)
        with self.assertRaises(AssertionError):
            dynamic.getObjectByIndex(8)


class TestObjectCollection(unittest.TestCase):

    def setUp(self) -> None:
        self.dummy = Circle(np.array([0, 0]), r=3)
        self.staticA = StaticObject(self.dummy)
        self.staticB = StaticObject(self.dummy)
        self.dynamicA = DynamicObject()
        self.dynamicA.addObject(self.dummy, 0)
        self.dynamicB = DynamicObject()
        self.dynamicB.addObject(self.dummy, 0)

    def test_add(self):
        # static plus static
        self.assertTrue(isinstance(self.staticA | self.staticB, ObjectCollection))
        self.assertTrue(isinstance(self.staticB | self.staticA, ObjectCollection))
        # dynamic + dynamic
        self.assertTrue(isinstance(self.dynamicA | self.dynamicB, ObjectCollection))
        self.assertTrue(isinstance(self.dynamicB | self.dynamicA, ObjectCollection))
        # static / dynamic mixed
        self.assertTrue(isinstance(self.staticA | self.dynamicA, ObjectCollection))
        self.assertTrue(isinstance(self.dynamicA | self.staticB, ObjectCollection))
        # object collection plus every other type
        collection = self.staticA | self.dynamicA
        self.assertTrue(isinstance(collection | self.staticB, ObjectCollection))
        self.assertTrue(isinstance(collection | self.dynamicB, ObjectCollection))
        self.assertTrue(isinstance(collection | collection, ObjectCollection))

    def test_len(self):
        # add same object
        self.assertEqual(len(self.staticA | self.staticA), 1)
        self.assertEqual(len(self.dynamicA | self.dynamicA), 1)

        # add different objects
        self.assertEqual(len(self.staticA | self.staticB), 2)
        self.assertEqual(len(self.dynamicA | self.staticB), 2)
        self.assertEqual(len(self.dynamicA | self.staticB | self.staticA), 3)

    def test_sub(self):
        # remove static
        a = self.staticA | self.staticB | self.dynamicA
        a -= self.staticB
        # check if only two elements are in the set
        self.assertEqual(len(a), 2)
        # check if correct objects are in the set
        self.assertTrue(self.staticA in a.objects)
        self.assertFalse(self.staticB in a.objects)
        self.assertTrue(self.dynamicA in a.objects)

        # remove object not in the set
        a -= self.dynamicB
        # check if only two elements are in the set
        self.assertEqual(len(a), 2)
        self.assertFalse(self.dynamicB in a.objects)

    def test_intersection(self):
        a = self.staticA | self.staticB | self.dynamicA
        b = self.dynamicB | self.dynamicA

        # check intersection of a and b
        self.assertEqual(len(a & b), 1)
        self.assertTrue(self.dynamicA in (a & b).objects)

        # check multi element intersection
        b |= self.staticA
        self.assertEqual(len(a & b), 2)
        self.assertTrue(self.dynamicA in (a & b).objects)
        self.assertTrue(self.staticA in (a & b).objects)

    def test_get_objects(self):
        unit_square = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])

        dyn1 = DynamicObject()
        pc1 = list()
        for i in range(5):
            pc1.append(PolygonCollection({Polygon(unit_square + np.array([i * 5, 0]))}))
            dyn1.addObject(pc1[-1], i)
        dyn2 = DynamicObject()
        pc2 = list()
        for i in range(5):
            pc2.append(PolygonCollection({Polygon(unit_square + np.array([i * 5, 5]))}))
            dyn2.addObject(pc2[-1], i)
        dyn3 = DynamicObject()
        pc3 = list()
        for i in range(5):
            pc3.append(PolygonCollection({Polygon(unit_square + np.array([i * 5, -5]))}))
            dyn3.addObject(pc3[-1], i)

        collection: ObjectCollection = dyn1 | dyn2

        # retrieve and check if the objects overlap
        for i in range(5):
            c = collection.getObject(i)
            gt = pc1[i] | pc2[i]
            # result is 2, since the squares penetration depth is 2
            self.assertEqual(c.overlap(gt), 2)
            self.assertEqual(dyn3.getObjectByIndex(-5 + i).overlap(pc3[i]), 2)


if __name__ == '__main__':
    unittest.main()
