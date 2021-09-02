from abc import *
from collections import OrderedDict
from enum import Enum
from typing import Set

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import shapely.affinity as af
import shapely.geometry as sh

_DEBUG = False


class IColor(Enum):
    N = 0
    R = 1
    G = 2
    B = 3


class SpatialInterface(ABC):
    """
    Interface for spatial relation logic. All objects need to provide a quantitative semantic.
    """

    @abstractmethod
    def shapes(self) -> set:
        """
        Returns the shapes stored in the SpatialInterface object
        Returns: The shapes of the SpatialInterface object

        """
        pass

    @abstractmethod
    def distance(self, other: 'SpatialInterface') -> float:
        """
        Returns the signed distance to another spatial interface object
        Args:
            other: The other spatial interface object

        Returns: Distance (squared) to other object

        """
        pass

    @abstractmethod
    def overlap(self, other: 'SpatialInterface') -> float:
        """
        Computes if this object overlaps with another object
        Args:
            other: The other object

        Returns: >=0 if both objects overlap and <0 otherwise

        """
        pass

    @abstractmethod
    def enclosed_in(self, other: 'SpatialInterface') -> float:
        """
        Computes if this objects is enclosed in another object. If any this object is a collection, every object
        must be enclosed in an object of other
        Args:
            other: The other object

        Returns: >=0 if this object is enclosed in the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def proximity(self, other: 'SpatialInterface', eps: float) -> bool:
        """
        Computes if this objects is in proximity to another object
        Args:
            other: The other object
            eps: Specification of proximity

        Returns: >=0 if objects are in proximity and <0 otherwise

        """
        pass

    @abstractmethod
    def distance_compare(self, other: 'SpatialInterface', eps: float, fun) -> bool:
        """
        Compares the distance between two objects and a target value (e.g., a dist b <= eps)
        Args:
            other: The other object
            eps: The target value
            fun: The function for comparing (<=,>=,==)

        Returns: >=0 if predicate is true and <0 otherwise

        """
        pass

    @abstractmethod
    def touching(self, other: 'SpatialInterface') -> bool:
        """
        Computes if two objects are touching
        Args:
            other: The other object

        Returns:

        """
        pass

    @abstractmethod
    def angle(self, other: 'SpatialInterface') -> bool:
        """
        Computes the angle between to objects
        Args:
            other: The other object

        Returns: NOT YET IMPLEMENTED / USED

        """
        pass

    @abstractmethod
    def above(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is above another object
        Args:
            other: The other object

        Returns: >= 0 if this object is above the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def below(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is below another object
        Args:
            other: The other object

        Returns: >= 0 if this object is below the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def left_of(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is left of another object
        Args:
            other: The other object

        Returns: >= 0 if this object is left of the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def right_of(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is right of another object
        Args:
            other: The other object

        Returns: >= 0 if this object is right of the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def close_to(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is close to another object
        Args:
            other: The other object

        Returns: >= 0 if this object is close to the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def far_from(self, other: 'SpatialInterface') -> bool:
        """
        Computes if this object is far from another object
        Args:
            other: The other object

        Returns: >= 0 if this object is far from the other object and <0 otherwise

        """
        pass

    @abstractmethod
    def closer_to_than(self, closer: 'SpatialInterface', than: 'SpatialInterface') -> bool:
        """
        Computes if this object is closer to one object than another
        Args:
            closer: The object that should be closer
            than: The object that should be further away

        Returns: >= 0 if this object is closer to one object than another and <0 otherwise

        """
        pass

    @abstractmethod
    def enlarge(self, radius: float) -> 'SpatialInterface':
        """
        Enlarges an object with a given radius
        Args:
            radius: The radius for enlarging the object

        Returns: The enlarged object

        """
        pass

    @abstractmethod
    def __or__(self, other: 'SpatialInterface'):
        pass

    @abstractmethod
    def __sub__(self, other: 'SpatialInterface'):
        pass


class ObjectInTime(ABC):
    """
    Interface for an object changing with time.
    """

    @abstractmethod
    def getObject(self, time) -> 'SpatialInterface':
        """
        Returns the object at the given time point
        """
        pass

    @abstractmethod
    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        """
        Returns the object at the given time point (given as index)
        Args:
            idx: The index of the time step

        Returns:

        """
        pass


class DynamicObject(ObjectInTime):

    def __init__(self):
        self._shapes = OrderedDict()  # compatible with all Python versions, preserves insertion order
        self._latest_time = None

    def addObject(self, object: SpatialInterface, time: int):
        if self._latest_time is None:
            self._latest_time = time - 1
        assert time not in self._shapes, '<DynamicObject/add>: time step already added! t={}'.format(time)
        assert time == self._latest_time + 1, '<DynamicObject/add>: time step missing! t = {}'.format(time)
        self._shapes[time] = object
        self._latest_time = time

    def getObject(self, time) -> 'SpatialInterface':
        assert time in self._shapes, '<DynamicObject/add>: time step not yet added! t={}'.format(time)
        return self._shapes[time]

    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        assert idx < len(self._shapes) if idx >= 0 else abs(idx) <= len(self._shapes)
        return list(self._shapes.values())[idx]

    def __or__(self, other):
        if isinstance(other, (StaticObject, DynamicObject)):
            return ObjectCollection(self, other)
        elif isinstance(other, ObjectCollection):
            return other + self
        else:
            raise Exception('<DynamicObject/add>: Provided object not supported! other = {}'.format(other))


class StaticObject(ObjectInTime):
    """
    An SpatialInterface object static in time. The simplest implementation of ObjectInTime
    """

    def __init__(self, spatial_object: SpatialInterface):
        super().__init__()
        self._spatial_obj = spatial_object

    def getObject(self, time) -> 'SpatialInterface':
        return self._spatial_obj

    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        return self._spatial_obj

    def __or__(self, other):
        if isinstance(other, (StaticObject, DynamicObject)):
            return ObjectCollection(self, other)
        elif isinstance(other, ObjectCollection):
            return other + self
        else:
            raise Exception('<DynamicObject/add>: Provided object not supported! other = {}'.format(other))


class ObjectCollection(ObjectInTime):

    def __init__(self, *args):
        self._object_set = set(args)

    def getObject(self, time) -> 'SpatialInterface':
        objs = [o.getObject(time) for o in self._object_set]
        shapes = [o.shapes() for o in objs]
        shapes = shapes[0].union(*shapes[1:])
        assert len(objs) > 0
        return type(objs[0])(shapes)

    def getObjectByIndex(self, idx: int) -> 'SpatialInterface':
        objs = [o.getObjectByIndex(idx) for o in self._object_set]
        shapes = [o.shapes for o in objs]
        shapes = shapes[0].union(*shapes[1:])
        assert len(objs) > 0
        return type(objs[0])(shapes)

    def __len__(self):
        return len(self._object_set)

    @property
    def objects(self):
        return self._object_set

    def __or__(self, other):
        collection = ObjectCollection()
        if isinstance(other, ObjectCollection):
            collection._object_set = self._object_set | other._object_set
        elif isinstance(other, (StaticObject, DynamicObject)):
            collection._object_set = self._object_set | {other}
        else:
            raise Exception('<ObjectCollection/add>: Provided object not supported! other = {}'.format(other))
        return collection

    def __sub__(self, other):
        collection = ObjectCollection()
        if isinstance(other, ObjectCollection):
            collection._object_set = self._object_set - other._object_set
        elif isinstance(other, (StaticObject, DynamicObject)):
            collection._object_set = self._object_set - {other}
        else:
            raise Exception('<ObjectCollection/add>: Provided object not supported! other = {}'.format(other))
        return collection

    def __and__(self, other):
        collection = ObjectCollection()
        if isinstance(other, ObjectCollection):
            collection._object_set = self._object_set & other._object_set
        elif isinstance(other, (StaticObject, DynamicObject)):
            collection._object_set = self._object_set & {other}
        else:
            raise Exception('<ObjectCollection/add>: Provided object not supported! other = {}'.format(other))
        return collection


class Polygon(object):
    """
    Class representing a polygon
    """
    _id = 0
    _ORIGIN = sh.Point([0, 0])  # origin for penetration depth computation
    # _MinkowskiDiff = lambda a, b: sh.Polygon(np.vstack(np.repeat([a],len(b),axis=0)-b)).convex_hull
    _MinkowskiDiff = lambda a, b: sh.Polygon(np.vstack([a - v for v in b])).convex_hull

    @classmethod
    def _get_id(cls):
        cls._id += 1
        return cls._id

    def __init__(self, vertices: np.ndarray, color: IColor = IColor.N, convex_hull: bool = True):
        """
        Initializes a polygon object
        Args:
            vertices: The vertices of the polygon
            color: The color of the polygon. Default = IColor.N
            convex_hull: bool to set if convex hull should be computed. Default = True
        """

        assert isinstance(vertices, np.ndarray), '<Polygon/init>: vertices must be of type np.ndarray!'

        if convex_hull:
            self.shape = sh.Polygon(vertices).convex_hull
        else:
            self.shape = sh.Polygon(vertices)
        self.color = color
        self.id = self._get_id()

    @property
    def shape(self) -> sh.Polygon:
        """
        Returns the shapely polygon object of this polygon
        Returns: Shapely polygon

        """
        return self._shape

    @shape.setter
    def shape(self, shape: sh.Polygon):
        """
        Sets the shapely polygon of this polygon
        Args:
            shape: The new shapely polygon

        """
        assert isinstance(shape, sh.Polygon), '<Polygon/shape>: Only shapely polygons are supported'
        self._shape = shape

    @property
    def vertices(self) -> np.ndarray:
        """
        Returns the vertices of the polygon
        Returns: The vertices of the polygon as a numpy array

        """
        return np.array(self.shape.exterior.coords)

    @property
    def center(self) -> np.ndarray:
        """
        Returns the geometric center of the polygon
        Returns: Geometric center of the polygon as a numpy array

        """
        return np.array(self.shape.centroid)

    def enlarge(self, radius: float) -> 'Polygon':
        enlarged = self.shape.buffer(radius)
        return Polygon(np.array(enlarged.exterior.coords))

    def translate(self, t: np.ndarray):
        """
        Translates the polygon by the given translation vector
        Args:
            t: Translation vector as numpy array with shape (2x1)

        Returns: Translated version of this polygon (no copy)

        """

        assert len(t) == 2
        self.shape = af.translate(self.shape, t[0], t[1])
        return self

    def rotate(self, theta: float, from_origin: bool = True, use_radians=False):
        """
        Rotates the polygon around its center (of its bounding box)
        Args:
            theta: The angle of the rotation
            from_origin: currently not used
            use_radians: True if angle is given in radian

        Returns: Rotated version of this polygon (no copy)

        """

        self.shape = af.rotate(self.shape, theta, origin='center', use_radians=use_radians)
        return self

    def distance(self, other: 'Polygon'):
        """
        Computes the distance to another polygon object
        Args:
            other: The other polygon object

        Returns: The distance (>=0) between this and the other object

        """
        return self.shape.distance(other.shape)

    def penetration_depth(self, other: 'Polygon'):
        """
        Computes the penetration depth with another polygon object
        Args:
            other: The other polygon object

        Returns: The penetration depth (>=0) between this and the other object.
        Zero if no intersection between the objects.

        """
        # return Polygon._MinkowskiDiff(np.asarray(self.shape.exterior.coords),
        #                              np.asarray(other.shape.exterior.coords)).exterior.distance(self._ORIGIN)
        return self._penetration_depth(np.asarray(self.shape.exterior.coords), np.asarray(other.shape.exterior.coords))

    def _penetration_depth(self, vert1: np.ndarray, vert2: np.ndarray):
        return Polygon._MinkowskiDiff(vert1, vert2).exterior.distance(self._ORIGIN)

    def signed_distance(self, other: 'Polygon'):
        """
        Computes the signed distance of this polygon to another one
        Args:
            other: The other polygon

        Returns: The signed distance between the two polygons (<= 0 if touching/intersection, >0 if no penetration)

        """
        gjk = self.distance(other)
        return gjk - self.penetration_depth(other) if gjk <= 0.0000001 else gjk

    def enclosedIn(self, other: 'Polygon'):
        """
        Computes if this polygon is enclosed in another polygon (i.e., all vertices are have negative signed distance)
        Args:
            other: The superset polygon

        Returns: >=0 if this polygon is enclosed in the other polygon, <0 otherwise

        """
        sd = -np.inf
        o = np.array(other.shape.exterior.coords)
        for v in self.vertices:
            gjk = other.shape.distance(sh.Point(v))
            sd_c = gjk - self._penetration_depth(v, o) if gjk < 0.0000001 else gjk
            if sd_c > sd:
                sd = sd_c
        return -sd if not np.isclose(sd, 0) else sd

    def contains_point(self, point: np.ndarray):
        """
        Checks whether a given point is enclosed in the polygon
        Args:
            point: The point to check

        Returns: True if the point is enclosed in the polygon and False otherwise

        """
        return self.shape.contains(sh.Point(point))

    @property
    def color(self) -> IColor:
        """
        Color of polygon
        Returns: Color of polygon

        """
        return self._color

    @color.setter
    def color(self, color: IColor):
        """
        Color of polygon
        Args:
            color: New color of circle

        """
        self._color = color

    def plot(self, ax=None, alpha=1.0, label: bool = True, color='k'):
        """
        Plots the polygon
        Args:
            ax: The axis object to plot to (if provided)
            alpha: The alpha value of the circle
            label: bool to indicate whether to plot label

        """

        if ax is None:
            ax = plt.gca()

        ax.add_patch(
            mp.Polygon(self.vertices, color=color, alpha=alpha))
        if label:
            plt.text(self.center[0], self.center[1], s=str(self.id), c='white', bbox=dict(facecolor='white', alpha=0.5))

    def minkowski_sum(self, other: 'Polygon', sub: bool = False) -> 'Polygon':
        new_vertices = list()
        for v in other.vertices:
            if not sub:
                new_vertices.append(self.vertices + (v - other.center))
            else:
                new_vertices.append(self.vertices - (v - other.center))
        return Polygon(np.vstack(new_vertices))

    def __add__(self, other):
        return self.minkowski_sum(other)

    def __sub__(self, other):
        return self.minkowski_sum(other, sub=True)

    def __hash__(self):
        return self.id


class Circle(Polygon):

    def __init__(self, center: np.ndarray, r: float):
        # approximate circle
        vertices = np.array(sh.Point(center).buffer(r).exterior)
        super().__init__(vertices, convex_hull=False)


class PolygonCollection(SpatialInterface):
    """
        Implements spatial interface for objects of type polytope. Represents set of polytopes
        """

    def __init__(self, polygons: Set[Polygon]):
        """
        Initializes a circle collection with a set of circles
        Args:
            circles: Set of circles
        """
        self.polygons = polygons if isinstance(polygons, set) else set(polygons)

    @property
    def polygons(self) -> Set[Polygon]:
        """
        Set of polygons
        Returns: set of polytopes

        """
        return self._polygons

    @polygons.setter
    def polygons(self, polygons: Set[Polygon]):
        """
        Set of polygons
        Args:
            polygons: new set of polytopes

        Returns:

        """
        self._polygons = polygons

    def add(self, p: Polygon):
        """
        Adds a polygons object to this collection
        Args:
            p: The polygons to add


        """
        self.polygons.add(p)

    def remove(self, p: Polygon):
        """
        Removes a polygons from this collection
        Args:
            p: The polygons to remove


        """
        self.polygons.discard(p)

    def shapes(self) -> set:
        return self.polygons

    def of_color(self, color: IColor) -> 'PolygoneCollection':
        """
        Returns a polygons collection containing polytopes of the specified color
        Args:
            color: The specified color

        Returns: polygons collection containing polytopes of specific color

        """
        return PolygonCollection(set([p for p in self.polygons if p.color == color]))

    def plot(self, ax=None, color='k', label=True):
        """
        Draws all polygons in this collection
        Args:
            ax: The axis object to plot to
            label: bool to indicate whether to plot labels

        Returns:

        """
        if ax is None:
            ax = plt.gca()
        for p in self.polygons:
            p.plot(ax=ax, label=label, color=color)
        plt.autoscale()
        plt.axis('equal')

    def distance(self, other: 'SpatialInterface') -> float:
        assert isinstance(other, PolygonCollection), \
            '<Polygon/distance>: Other object must be of type polygon, got {}'.format(other)

        # compute distances

        result = list()
        for p in self.polygons:
            result.append([p.signed_distance(o) for o in other.polygons])

        return result

    def overlap(self, other: 'SpatialInterface') -> bool:
        # intersection polygons
        inter = list()
        for p in self.polygons:
            inter.append([-p.signed_distance(o) for o in other.polygons])
        inter = np.array(inter)

        return np.max(inter)

    def enclosed_in(self, other: 'SpatialInterface') -> float:
        enclosed = list()
        for p in self.polygons:
            enclosed.append(np.array([p.enclosedIn(o) for o in other.polygons]).max())
        return np.array(enclosed).min()

    def proximity(self, other: 'SpatialInterface', eps: float) -> bool:
        return self.distance_compare(other, eps, np.less_equal)

    def distance_compare(self, other: 'SpatialInterface', eps: float, fun):
        assert np.positive(eps), '<Polygon>: Epsilon must be positive, got {}'.format(eps)

        # compute result
        if fun == np.less_equal:
            return np.max(np.repeat(eps, len(other.polygons)) - self.distance(other))
        if fun == np.greater_equal:
            return np.max(self.distance(other) - np.repeat(eps, len(other.polygons)))
        if fun == np.equal:
            return np.min([np.max(np.repeat(eps, len(other.polygons)) - self.distance(other)),
                           np.max(self.distance(other) - np.repeat(eps, len(other.polygons)))])

    def touching(self, other: 'SpatialInterface', eps: float = 5) -> bool:
        return self.proximity(other, eps=eps)
        return np.min([self.proximity(other, eps=eps), -self.proximity(other, eps=-eps)])

    def _min(self, axis: int) -> float:
        """
        Returns the minimum value of the projection of all polygons to the specified axis
        Args:
            axis: The specified axis

        Returns: The minimum value along the specified axis

        """
        return np.min([c.center[axis] for c in self.polygons])

    def _max(self, axis: int) -> float:
        """
        Returns the maximum value of the projection of all polygons to the specified axis
        Args:
            axis: The specified axis

        Returns: The maximum value along the specified axis

        """
        return np.max([c.center[axis] for c in self.polygons])

    def left_of(self, other: 'SpatialInterface') -> float:
        return other._min(0) - self._max(0)

    def right_of(self, other: 'SpatialInterface') -> float:
        return self._min(0) - other._max(0)

    def above(self, other: 'SpatialInterface') -> float:
        return self._min(1) - other._max(1)

    def below(self, other: 'SpatialInterface') -> float:
        return other._min(1) - self._max(1)

    def close_to(self, other: 'SpatialInterface') -> float:
        return self.proximity(other, 70.)

    def far_from(self, other: 'SpatialInterface') -> float:
        return -self.proximity(other, 150)

    def closer_to_than(self, closer: 'SpatialInterface', than: 'SpatialInterface') -> float:
        return np.min(self.distance(than)) - np.min(self.distance(closer))

    def enlarge(self, radius: float) -> 'SpatialInterface':
        return PolygonCollection(set([p.enlarge(radius) for p in self.polygons]))

    def angle(self, other: 'CircleOLD') -> float:
        pass

    def __or__(self, other: 'PolytopeCollection'):
        return PolygonCollection(self.polygons | other.polygons)

    def __sub__(self, other: 'PolytopeCollection'):
        return PolygonCollection(self.polygons - other.polygons)


if __name__ == '__main__':

    p1 = Polygon(np.array([[0, 0], [3, 3], [6, 0]]))
    p2 = Polygon(np.array([[3, 5], [7, 8], [10, 6]]))
    p2 = p2.rotate(30.45)
    p3 = Polygon(np.array([[3, 5], [7, 8], [10, 6]]) - 4, IColor.B)

    p_sum = p1.minkowski_sum(p2)
    (p1 + p2).plot(color='r')
    (p1 - p2).plot(color='g')
    plt.autoscale()
    plt.show()
    print(p1)

    p1.plot()
    p2.plot()
    p3.plot()
    plt.autoscale()
    plt.show()
    import time

    a = sh.Polygon(p1.vertices)
    b = sh.Polygon(p2.vertices)
    print('Area is {}'.format(a.area))
    print('Distance is {}'.format(a.distance(b)))

    t0 = time.time()
    for i in range(100):
        a.distance(b)
    print(f'Time took {time.time() - t0}')

    pc = PolygonCollection(set([p1, p2]))
    pd = PolygonCollection(set([p3]))

    t0 = time.time()
    print('Distance is {}'.format(pc.distance(pd)))
    print('Distance is {}'.format(pc.distance(pc)))
    print('Intersecting is {}'.format(pc.overlap(pd)))
    print('Intersecting is {}'.format(pc.overlap(pc)))
    print(f'Time took {time.time() - t0}')

    t0 = time.time()
    for i in range(1):
        pc.distance(pc)
        # p1.intersect(p2).volume
    print(f'Time took for 10 {time.time() - t0}')
