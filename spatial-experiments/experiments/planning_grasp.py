import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from lark import Token
from matplotlib import cm

from sim.grasp.lib import PandaGraspEnv, update_ANGLES, min_ray, max_ray
from spatial_spec.automaton_planning import AutomatonPlanner
from spatial_spec.geometry import Circle, Polygon, PolygonCollection, StaticObject
from spatial_spec.logic import Spatial


class GraspableObject:

    def __init__(self, object_id, name, position, orientation, shape_info):
        self.id = object_id
        self.name = name
        self.pos = position
        self.ori = orientation
        self.shape_info = shape_info
        self.angle = 0

    def get_shape(self):
        if self.shape_info[0] == "circle":
            return Circle(self.pos, self.shape_info[1])
        if self.shape_info[0] == "rect":
            return Polygon(rectangle_around_center(self.pos[:2], self.shape_info[1][0], self.shape_info[1][1])).rotate(self.angle, use_radians=True)
        raise ValueError("Unexpected shape info in graspable object!")

    def get_static_shape(self):
        return StaticObject(PolygonCollection({self.get_shape()}))


def get_positions(obj_list):
    pos_arr = []
    for go in obj_list:
        pos_arr.append(go.pos)
    return pos_arr


def get_orientation(obj_list):
    ori_arr = []
    for go in obj_list:
        ori_arr.append(go.ori)
    return ori_arr


def get_angles(obj_list):
    ang_arr = []
    for go in obj_list:
        ang_arr.append(go.angles)
    return ang_arr


def update_angles(obj_list):
    pos_arr = get_positions(obj_list)
    ori_arr = get_orientation(obj_list)
    ang_arr = update_ANGLES(pos_arr, ori_arr)

    for i, angle in enumerate(ang_arr):
        obj_list[i].angle = angle


def update_objects(grasp_obj_list, pos_arr, ori_arr, angles_arr):
    for i, go in enumerate(grasp_obj_list):
        go.pos = pos_arr[i]
        go.ori = ori_arr[i]
        go.angle = angles_arr[i]


def rectangle_around_center(center: np.ndarray, box_length1: float, box_length2: float) -> np.ndarray:
    return np.array(
        [center + [-box_length1 / 2, -box_length2 / 2],
         center + [box_length1 / 2, -box_length2 / 2],
         center + [box_length1 / 2, box_length2 / 2],
         center + [-box_length1 / 2, box_length2 / 2]])


def observation(spatial_interpreter, spatial_vars, ap_list):
    obs = ''
    for var_ap in ap_list:
        subtree = spatial_vars[var_ap]
        if spatial_interpreter.interpret(subtree) > 0:
            obs += '1'
        else:
            obs += '0'
    return obs


def gradient_map(spatial_interpreter, spatial_tree, graspable_object):
    positions = np.c_[gx.ravel(), gy.ravel()]
    virtual_object = copy.deepcopy(graspable_object)

    grad_values = []
    # compute values
    for position in positions:
        # move the virtual_object
        virtual_object.pos = position
        spatial_interpreter.reset_spatial_dict()
        spatial_interpreter.assign_variable(virtual_object.name, virtual_object.get_static_shape())
        grad_values.append(spatial.interpret(spatial_tree))

    # reset the object position
    spatial_interpreter.reset_spatial_dict()
    spatial_interpreter.assign_variable(graspable_object.name, graspable_object.get_static_shape())

    return grad_values


def get_composite_constraint_map(spatial_interpreter, spat_var_dict, dfa_ap, object_to_move, constraints):
    result = []
    for constraint in constraints:
        constraint_map = gradient_map_from_guard(spatial_interpreter, spat_var_dict, dfa_ap, object_to_move, guard=constraint)

        # merge constraint map into composite constraint map
        # since we don't want to satisfy any constraint, we simply remember the maximum
        if len(result) > 0:
            result = np.minimum(constraint_map, result)
        else:
            result = constraint_map

    return result


def gradient_map_from_guard(spatial_interpreter, spat_var_dict, dfa_ap, object_to_move, guard):
    result = []
    for i, guard_val in enumerate(guard):
        # skip don't care variables
        if guard_val == 'X':
            continue

        tree = spat_var_dict[dfa_ap[i]]
        gradient_values = gradient_map(spatial_interpreter, tree, object_to_move)

        # if the guard has the variable as negative, flip the gradient map
        if guard_val == '0':
            gradient_values = [-1 * x for x in gradient_values]

        # merge results into the constraint_map (by logical conjunction)
        if len(result) > 0:
            result = np.minimum(result, gradient_values)
        else:
            result = gradient_values

    return result


def remove_obstacle_from_gradient(gradient, obj):
    positions = np.c_[gx.ravel(), gy.ravel()]
    for i, position in enumerate(positions):
        if obj.contains_point(position):
            gradient[i] = np.nan
    return gradient


def find_best_point(map_2d, threshold):
    boolean_table = map_2d > threshold

    # forbid all positions that are constrained
    for i in range(map_2d.shape[0]):
        for j in range(map_2d.shape[1]):
            if np.isnan(map_2d[i][j]):
                boolean_table[i, j] = False

    # forbid all positions that are to close or too far from the grasping arm
    for i in range(map_2d.shape[0]):
        for j in range(map_2d.shape[1]):
            dist_from_origin = rx[j] ** 2 + ry[i] ** 2
            if dist_from_origin < min_ray ** 2 or dist_from_origin > max_ray ** 2:
                boolean_table[i, j] = False

    # copy the gradient, mask the values
    masked_map_2d = np.array(map_2d, copy=True)
    for i in range(masked_map_2d.shape[0]):
        for j in range(masked_map_2d.shape[1]):
            if not boolean_table[i, j]:
                masked_map_2d[i, j] = np.nan

    if not np.any(masked_map_2d > 0):
        return None

    result = np.where(masked_map_2d == np.nanmax(masked_map_2d))
    # zip the 2 arrays to get the exact coordinates
    list_of_coordinates = list(zip(result[0], result[1]))
    id_x = list_of_coordinates[0][0]
    id_y = list_of_coordinates[0][1]

    return np.array([rx[id_y], ry[id_x], 0])


def get_relevant_objects(targets, dfa_ap, spat_vars):
    relv_objs = set()

    for trgt in targets:
        for i, bit in enumerate(trgt):
            # skip don't care bits
            if bit == 'X':
                continue
            # otherwise, it's relevant, so we get the subtree
            subtree = spat_vars[dfa_ap[i]]
            # get all leaves that correspond to a variable
            for token in subtree.scan_values(lambda x: isinstance(x, Token)):
                if token.type == 'NAME':
                    relv_objs.add(token.value)

    # only these objects can be picked up
    pickable = ["bottle", "banana", "mug", "gelatin", "sugarbox", "can", "crackerbox", "kanelbulle"]
    return relv_objs.intersection(pickable)

def main():
    # spatial interpreter
    spatial = Spatial(quantitative=True)
    # automaton-based planner
    planner = AutomatonPlanner()
    # pybullet grasping simulation
    sim = PandaGraspEnv(1)

    # specification
    spec = "(F(kanelbulle enclosedin plate))"
    spec += "& (F((banana dist plate <= 0.3) & (banana dist plate >= 0.1) & (banana leftof plate) & (banana below plate)))"
    spec += "& (F((mug dist plate <= 0.3) & (mug dist plate >= 0.1) & (mug leftof plate) & (mug above plate)))"
    spec += "& (F((bottle dist plate <= 0.3) & (bottle dist plate >= 0.1) & (bottle leftof plate) & (bottle above plate)))"
    spec += "& (F((sugarbox dist plate >= 0.4) & (sugarbox dist crackerbox <= 0.2)))"
    spec_tree = spatial.parse(spec)  # build the spatial tree
    planner.tree_to_dfa(spec_tree)  # transform the tree into an automaton

    # print the corresponding LTLf formula
    print("\nTemporal Structure:", planner.temporal_formula)
    print("Planner DFA Size:", len(planner.dfa.nodes), len(planner.dfa.edges), "\n")

    # parameters
    step_size = 0.1
    x_range = [-max_ray, max_ray]
    y_range = [-max_ray, max_ray]

    # gradient grid mesh
    rx, ry = np.arange(x_range[0], x_range[1], step_size), np.arange(y_range[0], y_range[1], step_size)
    gx, gy = np.meshgrid(rx, ry)

    # statistics
    counter_fallen = 0  # number of times an object has fallen
    counter_out = 0  # number of times an object is out of the reachable region
    action_fail = 0  # number of fails in reaching the target position
    counter_actions = 0  # number of actions performed

    # object initialization
    h = sim.get_table_height() + 0.08

    graspable_objects = [
        GraspableObject(object_id=0,
                        name='bottle',
                        position=np.array([-0.15, 0.5, h]),
                        orientation=p.getQuaternionFromEuler([math.pi / 2, 0, 0]),
                        shape_info=['circle', 0.024]),
        GraspableObject(object_id=1,
                        name='banana',
                        position=np.array([0.45, 0.23, h]),
                        orientation=p.getQuaternionFromEuler([0, 0, 0]),
                        shape_info=['rect', (0.05, 0.015)]),
        GraspableObject(object_id=2,
                        name='mug',
                        position=np.array([0.45, -0.2, h]),
                        orientation=p.getQuaternionFromEuler([0, 0, 0]),
                        shape_info=['circle', 0.03]),
        GraspableObject(object_id=3,
                        name='gelatin',
                        position=np.array([0.05, -0.55, h - 0.05]),
                        orientation=p.getQuaternionFromEuler([math.pi / 2, +math.pi / 2 + math.pi / 14, math.pi / 2]),
                        shape_info=['rect', (0.05, 0.025)]),
        GraspableObject(object_id=4,
                        name='sugarbox',
                        position=np.array([0.3, 0.4, h]),
                        orientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]),
                        shape_info=['rect', (0.05, 0.025)]),
        GraspableObject(object_id=5,
                        name='can',
                        position=np.array([-0.1, -0.45, h]),
                        orientation=p.getQuaternionFromEuler([0, 0, 0]),
                        shape_info=['circle', 0.03]),
        GraspableObject(object_id=6,
                        name='crackerbox',
                        position=np.array([0., -0.45, h]),
                        orientation=p.getQuaternionFromEuler([0, 0, math.pi / 2]),
                        shape_info=['rect', (0.05, 0.025)]),
        GraspableObject(object_id=7,
                        name='kanelbulle',
                        position=np.array([0.2, 0.55, h - .01]),
                        orientation=p.getQuaternionFromEuler([0, 0, 0]),
                        shape_info=['circle', 0.03]),
        GraspableObject(object_id=8,
                        name='plate',
                        position=np.array([0.4, 0., h]),
                        orientation=p.getQuaternionFromEuler([0, 0, 0]),
                        shape_info=['circle', 0.05])
    ]

    # object initialization - angles
    update_angles(graspable_objects)

    # object initialization - spatial variables
    for grasp_obj in graspable_objects:
        spatial.assign_variable(grasp_obj.name, grasp_obj.get_static_shape())

    # this dictionary contains a variable name to spatial tree mapping
    spatial_variables = planner.get_variable_to_tree_dict()

    # you have to define in which order you pass variable assignments
    trace_ap = list(spatial_variables.keys())

    # resets the automaton current state to the initial state (doesn't do anything here)
    planner.reset_state()

    # before you ask anything from the automaton, provide a initial observation of each spatial sub-formula
    init_obs = observation(spatial, spatial_variables, trace_ap)
    planner.dfa_step(init_obs, trace_ap)

    # load robot and object in the chosen pose
    print("Initializing Simulation... \n")
    sim.reset(get_positions(graspable_objects), get_orientation(graspable_objects), "../urdf")

    # planning loop
    while not planner.currently_accepting():
        target_set, constraint_set, edge = planner.plan_step()
        print("Considering", target_set, "with constraints", constraint_set, "...")
        # no path to accepting state exists, we're doomed
        if not target_set:
            print("It is impossible to satisfy the specification. Exiting planning loop...\n")
            break

        target_obj_id = None
        target_point = None

        # get all objects relevant to the current targets
        relevant_objects = get_relevant_objects(target_set, planner.get_dfa_ap(), spatial_variables)

        for obj_name in relevant_objects:
            print("Considering", obj_name, "...")
            # we already found a candidate and can directly execute
            if target_obj_id is not None:
                break

            # obtain the object information from the name
            relevant_obj = next(x for x in graspable_objects if x.name == obj_name)

            # compute composite constraint map
            composite_constraint_map = []
            if constraint_set:
                composite_constraint_map = get_composite_constraint_map(spatial_interpreter=spatial,
                                                                        spat_var_dict=spatial_variables,
                                                                        dfa_ap=planner.get_dfa_ap(),
                                                                        object_to_move=relevant_obj,
                                                                        constraints=constraint_set)

            # try out all target options
            for target in target_set:
                # get target map
                target_map = gradient_map_from_guard(spatial_interpreter=spatial,
                                                     spat_var_dict=spatial_variables,
                                                     dfa_ap=planner.get_dfa_ap(),
                                                     object_to_move=relevant_obj,
                                                     guard=target)

                # remove the composite constraint from the map
                if constraint_set:
                    assert len(target_map) == len(composite_constraint_map)
                    for v in range(len(target_map)):
                        if composite_constraint_map[v] > 0:
                            target_map[v] = np.nan

                # remove the objects from the map, but skip the plate (so we can put sth into the plate)
                # and skip the object we are moving
                inflated_shape = Circle(relevant_obj.pos, 0.08)
                for grasp_obj in graspable_objects:
                    if grasp_obj.name != "plate" or grasp_obj.name == relevant_obj.name:
                        target_map = remove_obstacle_from_gradient(target_map, grasp_obj.get_shape() - inflated_shape)

                # find the best point for the object
                target_point = find_best_point(np.array(target_map).reshape(gx.shape), threshold=0)
                # check if any positive values exist
                if target_point is not None:
                    # this object does it
                    target_obj_id = relevant_obj.id

                    # plot gradient values
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    values_2d = np.array(target_map).reshape(gx.shape)
                    granularity = 0.05
                    con = ax.contourf(gx, gy, values_2d,
                                      levels=np.arange(np.nanmin(values_2d) - granularity, np.nanmax(values_2d) + granularity, granularity),
                                      cmap=cm.coolwarm,
                                      alpha=0.3,
                                      antialiased=False)
                    # plot objects
                    for grasp_obj in graspable_objects:
                        grasp_obj.get_shape().plot(ax, label=False, color='r')
                    # plot target point
                    plt.plot(target_point[0], target_point[1], "ok")
                    plt.autoscale()
                    plt.colorbar(con)
                    plt.show()
                    break

        # this edge is completely impossible in this framework, we prune the edge from the automaton
        if target_obj_id is None:
            print("Chosen edge turned out to be impossible. Pruning the edge...\n")
            planner.dfa.remove_edge(edge[0], edge[1])
        else:
            # chose one of the target points and execute the grasp
            pos, ori, ang = sim.grasp(target_obj_id, target_point)

            # update position, orientation and angles
            update_objects(graspable_objects, pos, ori, ang)

            # update spatial variables
            for grasp_obj in graspable_objects:
                spatial.assign_variable(grasp_obj.name, grasp_obj.get_static_shape())

            # update automaton
            init_obs = observation(spatial, spatial_variables, trace_ap)
            planner.dfa_step(init_obs, trace_ap)

    print("Algorithm terminated. Specification satisfied:", planner.currently_accepting())
    # we are done, close the simulator
    sim.close()

if __name__ == '__main__':
    main()
