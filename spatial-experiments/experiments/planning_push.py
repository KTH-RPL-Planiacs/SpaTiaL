import copy
import heapq
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from lark import Token
from matplotlib import cm

from sim.push.lib import PandaPushEnv, minX, minY, maxX, maxY, update_pose
from spatial_spec.automaton_planning import AutomatonPlanner
from spatial_spec.geometry import Polygon, PolygonCollection, StaticObject
from spatial_spec.logic import Spatial


class PushableObject:

    def __init__(self, object_id, name, position, orientation, color):
        self.id = object_id
        self.name = name
        self.pos = position
        self.ori = orientation
        self.length = 0.032
        self.angle = 0
        self.color = color

    def get_shape(self):
        return Polygon(rectangle_around_center(self.pos[:2], self.length, self.length)).rotate(self.angle, use_radians=True)

    def get_static_shape(self):
        return StaticObject(PolygonCollection({self.get_shape()}))


def heuristic(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def blocked(c_x, c_y, d_x, d_y, matrix):
    if c_x + d_x < 0 or c_x + d_x >= matrix.shape[0]:
        return True
    if c_y + d_y < 0 or c_y + d_y >= matrix.shape[1]:
        return True
    if d_x != 0 and d_y != 0:
        if matrix[c_x + d_x][c_y] and matrix[c_x][c_y + d_y]:
            return True
        if matrix[c_x + d_x][c_y + d_y]:
            return True
    else:
        if d_x != 0:
            if matrix[c_x + d_x][c_y]:
                return True
        else:
            if matrix[c_x][c_y + d_y]:
                return True
    return False


def dblock(c_x, c_y, d_x, d_y, matrix):
    if matrix[c_x - d_x][c_y] and matrix[c_x][c_y - d_y]:
        return True
    else:
        return False


def direction(c_x, c_y, p_x, p_y):
    d_x = int(math.copysign(1, c_x - p_x))
    d_y = int(math.copysign(1, c_y - p_y))
    if c_x - p_x == 0:
        d_x = 0
    if c_y - p_y == 0:
        d_y = 0
    return d_x, d_y


def node_neighbours(c_x, c_y, parent, matrix):
    neighbours = []
    if type(parent) != tuple:
        for i, j in [
            (-1, 0),
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            if not blocked(c_x, c_y, i, j, matrix):
                neighbours.append((c_x + i, c_y + j))

        return neighbours
    d_x, d_y = direction(c_x, c_y, parent[0], parent[1])

    if d_x != 0 and d_y != 0:
        if not blocked(c_x, c_y, 0, d_y, matrix):
            neighbours.append((c_x, c_y + d_y))
        if not blocked(c_x, c_y, d_x, 0, matrix):
            neighbours.append((c_x + d_x, c_y))
        if (
                not blocked(c_x, c_y, 0, d_y, matrix)
                or not blocked(c_x, c_y, d_x, 0, matrix)
        ) and not blocked(c_x, c_y, d_x, d_y, matrix):
            neighbours.append((c_x + d_x, c_y + d_y))
        if blocked(c_x, c_y, -d_x, 0, matrix) and not blocked(
                c_x, c_y, 0, d_y, matrix
        ):
            neighbours.append((c_x - d_x, c_y + d_y))
        if blocked(c_x, c_y, 0, -d_y, matrix) and not blocked(
                c_x, c_y, d_x, 0, matrix
        ):
            neighbours.append((c_x + d_x, c_y - d_y))

    else:
        if d_x == 0:
            if not blocked(c_x, c_y, d_x, 0, matrix):
                if not blocked(c_x, c_y, 0, d_y, matrix):
                    neighbours.append((c_x, c_y + d_y))
                if blocked(c_x, c_y, 1, 0, matrix):
                    neighbours.append((c_x + 1, c_y + d_y))
                if blocked(c_x, c_y, -1, 0, matrix):
                    neighbours.append((c_x - 1, c_y + d_y))

        else:
            if not blocked(c_x, c_y, d_x, 0, matrix):
                if not blocked(c_x, c_y, d_x, 0, matrix):
                    neighbours.append((c_x + d_x, c_y))
                if blocked(c_x, c_y, 0, 1, matrix):
                    neighbours.append((c_x + d_x, c_y + 1))
                if blocked(c_x, c_y, 0, -1, matrix):
                    neighbours.append((c_x + d_x, c_y - 1))
    return neighbours


def jump(c_x, c_y, d_x, d_y, matrix, goal):
    n_x = c_x + d_x
    n_y = c_y + d_y
    if blocked(n_x, n_y, 0, 0, matrix):
        return None

    if (n_x, n_y) == goal:
        return n_x, n_y

    o_x = n_x
    o_y = n_y

    if d_x != 0 and d_y != 0:
        while True:
            if (
                    not blocked(o_x, o_y, -d_x, d_y, matrix)
                    and blocked(o_x, o_y, -d_x, 0, matrix)
                    or not blocked(o_x, o_y, d_x, -d_y, matrix)
                    and blocked(o_x, o_y, 0, -d_y, matrix)
            ):
                return o_x, o_y

            if (
                    jump(o_x, o_y, d_x, 0, matrix, goal) is not None
                    or jump(o_x, o_y, 0, d_y, matrix, goal) is not None
            ):
                return o_x, o_y

            o_x += d_x
            o_y += d_y

            if blocked(o_x, o_y, 0, 0, matrix):
                return None

            if dblock(o_x, o_y, d_x, d_y, matrix):
                return None

            if (o_x, o_y) == goal:
                return o_x, o_y
    else:
        if d_x != 0:
            while True:
                if (
                        not blocked(o_x, n_y, d_x, 1, matrix)
                        and blocked(o_x, n_y, 0, 1, matrix)
                        or not blocked(o_x, n_y, d_x, -1, matrix)
                        and blocked(o_x, n_y, 0, -1, matrix)
                ):
                    return o_x, n_y

                o_x += d_x

                if blocked(o_x, n_y, 0, 0, matrix):
                    return None

                if (o_x, n_y) == goal:
                    return o_x, n_y
        else:
            while True:
                if (
                        not blocked(n_x, o_y, 1, d_y, matrix)
                        and blocked(n_x, o_y, 1, 0, matrix)
                        or not blocked(n_x, o_y, -1, d_y, matrix)
                        and blocked(n_x, o_y, -1, 0, matrix)
                ):
                    return n_x, o_y

                o_y += d_y

                if blocked(n_x, o_y, 0, 0, matrix):
                    return None

                if (n_x, o_y) == goal:
                    return n_x, o_y


def identify_successors(c_x, c_y, came_from, matrix, goal):
    successors = []
    neighbours = node_neighbours(c_x, c_y, came_from.get((c_x, c_y), 0), matrix)

    for cell in neighbours:
        d_x = cell[0] - c_x
        d_y = cell[1] - c_y

        jump_point = jump(c_x, c_y, d_x, d_y, matrix, goal)

        if jump_point is not None:
            successors.append(jump_point)

    return successors


def jps(matrix, start, goal):
    came_from = {}
    close_set = set()
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}

    pqueue = []
    heapq.heappush(pqueue, (fscore[start], start))

    while pqueue:
        current = heapq.heappop(pqueue)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data

        close_set.add(current)

        successors = identify_successors(
            current[0], current[1], came_from, matrix, goal
        )

        for successor in successors:
            jump_point = successor

            if (
                    jump_point in close_set
            ):
                continue

            tentative_g_score = gscore[current] + length(current, jump_point)

            if tentative_g_score < gscore.get(
                    jump_point, 0
            ) or jump_point not in [j[1] for j in pqueue]:
                came_from[jump_point] = current
                gscore[jump_point] = tentative_g_score
                fscore[jump_point] = tentative_g_score + heuristic(
                    jump_point, goal
                )
                heapq.heappush(pqueue, (fscore[jump_point], jump_point))
    return None


def length(current, jump_point):
    return math.sqrt((current[0] - jump_point[0]) ** 2 + (current[1] - jump_point[1]) ** 2)


def total_length(path_list):
    total = 0
    prev = path_list[0]

    for current in path_list:
        if current == prev:
            continue
        total += length(current, prev)

    return total


def get_goal_candidates(v_2d, threshold):
    candidates = set()
    for x in range(len(v_2d)):
        for y in range(len(v_2d[x])):
            if np.isnan(v_2d[x][y]) or v_2d[x][y] <= threshold:
                continue
            # search neighbours
            for i, j in [
                (-1, 0),
                (0, -1),
                (1, 0),
                (0, 1),
            ]:
                nb_x = np.clip(x + i, 0, len(v_2d) - 1)
                nb_y = np.clip(y + j, 0, len(v_2d[x]) - 1)
                if not np.isnan(v_2d[nb_x][nb_y]) and v_2d[nb_x][nb_y] <= threshold:
                    candidates.add((y, x))
                    break

    return candidates


def pos_to_coords(p, range_x, range_y):
    posx = None
    posy = None

    for i in range(len(rx) - 1):
        if range_x[i + 1] > p[0]:
            posx = i
            break

    for i in range(len(ry) - 1):
        if range_y[i + 1] > p[1]:
            posy = i
            break

    return posx, posy


def rectangle_around_center(center: np.ndarray, box_length1: float, box_length2: float) -> np.ndarray:
    return np.array(
        [center + [-box_length1 / 2, -box_length2 / 2],
         center + [box_length1 / 2, -box_length2 / 2],
         center + [box_length1 / 2, box_length2 / 2],
         center + [-box_length1 / 2, box_length2 / 2]])


def update_objects(obj_list, pos_arr, ori_arr, ang_arr):
    for i, obj in enumerate(obj_list):
        obj_list[i].pos = pos_arr[i]
        obj_list[i].ori = ori_arr[i]
        obj_list[i].angle = ang_arr[i]


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
            result = np.maximum(constraint_map, result)
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
    return relv_objs


def plot_path(v_2d, obj_list, path_to_plot, start, goal_cand, plot_path=False, name=None):
    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 10}
    # matplotlib.rc('font', **font)
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['font.sans-serif'] = 'Latin Modern Math'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(3.5, 2.5))

    ax = fig.add_subplot(111)
    granularity = 0.02
    con = ax.contourf(gx, gy, v_2d,
                      levels=np.arange(np.nanmin(v_2d) - granularity, np.nanmax(v_2d) + granularity, granularity),
                      cmap=cm.coolwarm,
                      alpha=0.3,
                      antialiased=False)

    for obj in obj_list:
        obj.get_shape().plot(ax, label=False, color=obj.color)

    if plot_path:
        for g_x, g_y in goal_cand:
            plt.plot(rx[g_x], ry[g_y], "ok")

        plt.plot(rx[start[0]], ry[start[1]], "ok")

        for p_x, p_y in path_to_plot:
            plt.plot(rx[p_x], ry[p_y], "og")

    plt.xlabel('position [m]')
    plt.ylabel('position [m]')
    plt.autoscale()
    plt.colorbar(con)
    if name is not None:
        # save to file
        plt.savefig(name + '.svg')
    plt.show()


if __name__ == '__main__':
    # spatial interpreter
    spatial = Spatial(quantitative=True)
    # automaton-based planner
    planner = AutomatonPlanner()
    # pybullet pushing simulation
    sim = PandaPushEnv(1)

    # specification
    spec = "("
    spec += "(F((green rightof red) & (green rightof blue)))"
    spec += "& ((!t((green rightof red) & (green rightof blue))) U (red above blue))"
    spec += "& (G ((green dist red >= 0.03) & (green dist blue >= 0.03) & (red dist blue >= 0.03)))"
    spec += ")"

    spec_tree = spatial.parse(spec)  # build the spatial tree
    planner.tree_to_dfa(spec_tree)  # transform the tree into an automatonä
    print(planner.ltlf_parser.to_dot())
    print(planner.spatial_dict)

    # print the corresponding LTLf formula
    print("\nTemporal Structure:", planner.temporal_formula)
    print("Planner DFA Size:", len(planner.dfa.nodes), len(planner.dfa.edges), "\n")

    # parameters
    step_size = 0.005
    x_range = [minX, maxX]
    y_range = [minY, maxY]

    # gradient grid mesh
    rx, ry = np.arange(x_range[0], x_range[1], step_size), np.arange(y_range[0], y_range[1], step_size)
    gx, gy = np.meshgrid(rx, ry)

    # object initialization
    h = sim.get_table_height() + 0.08

    pushable_objects = [
        PushableObject(object_id=0,
                       name="green",
                       position=np.array([0.4, 0, h]),
                       orientation=p.getQuaternionFromEuler([0, 0, 0]),
                       color="g",
                       ),
        PushableObject(object_id=1,
                       name="red",
                       position=np.array([0.5, -0.05, h]),
                       orientation=p.getQuaternionFromEuler([0, 0, math.pi / 4]),
                       color="r",
                       ),
        PushableObject(object_id=2,
                       name="blue",
                       position=np.array([0.6, 0.15, h]),
                       orientation=p.getQuaternionFromEuler([0, 0, math.pi / 4]),
                       color="b",
                       ),
    ]

    # load robot and object in the chosen positions
    sim.reset(pushable_objects[0].pos,
              pushable_objects[1].pos,
              pushable_objects[2].pos,
              pushable_objects[0].ori,
              pushable_objects[1].ori,
              pushable_objects[2].ori,
              folder_path="../urdf")

    # get actual positions from the simulation ??? and update angles
    POS, OR, ANG = update_pose(sim._get_state())
    update_objects(pushable_objects, POS, OR, ANG)

    # object initialization - spatial variables
    for push_obj in pushable_objects:
        spatial.assign_variable(push_obj.name, push_obj.get_static_shape())

    # this dictionary contains a variable name to spatial tree mapping
    spatial_variables = planner.get_variable_to_tree_dict()

    # you have to define in which order you pass variable assignments
    trace_ap = list(spatial_variables.keys())

    # resets the automaton current state to the initial state (doesn't do anything here)
    planner.reset_state()

    # before you ask anything from the automaton, provide a initial observation of each spatial sub-formula
    init_obs = observation(spatial, spatial_variables, trace_ap)
    planner.dfa_step(init_obs, trace_ap)

    # planning loop
    while not planner.currently_accepting():
        target_set, constraint_set, edge = planner.plan_step()
        print("Considering", target_set, "with constraints", constraint_set, "(", planner.get_dfa_ap(), ")...")
        # no path to accepting state exists, we're doomed
        if not target_set:
            print("It is impossible to satisfy the specification. Exiting planning loop...\n")
            break

        target_obj_id = None
        target_path = None

        # get all objects relevant to the current targets
        relevant_objects = get_relevant_objects(target_set, planner.get_dfa_ap(), spatial_variables)

        for obj_name in relevant_objects:
            # we already found a candidate and can directly execute
            if target_obj_id is not None:
                break

            print("Considering", obj_name, "...")
            # obtain the object information from the name
            relevant_obj = next(x for x in pushable_objects if x.name == obj_name)

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

                values_2d = np.array(target_map).reshape(gx.shape)

                # check for a path
                # compute blocked matrix (true means cannot enter cell)
                blocked_matrix = np.isnan(values_2d).transpose()

                # compute start coords that fit the start position
                start_coords = pos_to_coords(relevant_obj.pos, rx, ry)

                # obtain goal candidates
                goal_candidates = get_goal_candidates(values_2d, threshold=0.01)

                # compute paths
                path_candidates = {}
                for goal_coords in goal_candidates:
                    # compute path
                    path = jps(blocked_matrix, start_coords, goal_coords)
                    if path is not None:
                        path_candidates[goal_coords] = {"path": path, "cost": total_length(path)}

                # at least one path exists
                if len(path_candidates) > 0:
                    best = min(path_candidates.items(), key=lambda x: x[1]["cost"])
                    shortest_path = best[1]["path"]

                    target_obj_id = relevant_obj.id
                    target_path = []
                    plot_path(values_2d, pushable_objects, shortest_path, start_coords, goal_candidates, plot_path=True, name=list(target_set)[0])
                    for path_x, path_y in shortest_path:
                        target_path.append(np.array([rx[path_x], ry[path_y], h]))
                    target_path.reverse()

        # this edge is completely impossible in this framework, we prune the edge from the automaton
        if target_obj_id is None:
            print("Chosen edge turned out to be impossible. Pruning the edge...\n")
            planner.dfa.remove_edge(edge[0], edge[1])
        else:
            # otherwise we can execute (pushing library does not need the starting position inside the path)
            sim.deletes_lines()
            sim.plot_path(target_path)
            sim.track(target_path, target_obj_id)
            sim.print_err()

            # get actual positions from the simulation ??? and update angles
            POS, OR, ANG = update_pose(sim._get_state())
            update_objects(pushable_objects, POS, OR, ANG)

            # update spatial variables
            for grasp_obj in pushable_objects:
                spatial.assign_variable(grasp_obj.name, grasp_obj.get_static_shape())

            # update automaton
            init_obs = observation(spatial, spatial_variables, trace_ap)
            planner.dfa_step(init_obs, trace_ap)

    print("Algorithm terminated. Specification satisfied:", planner.currently_accepting())
    # we are done, close the simulator
    sim.close()
