"""
Pybullet Franka Emika Robot - Pushing legos task

In this example for the given pose of the legos the gradient map is
computed; then the closest point in the map in which the satisfaction level
of the predefined spec is above the chosen threshold is found and using the
best-first-search algorithm a path to reach it is computed. Finally, the
robot trackes the path and pushes the lego to the desired position.
"""

import sys

from lib import *

sys.path.append("./../../scripts")

from gradient import generate_map
from bestfirst import best_first_search

###########
# Simulation


# Create the environment
env = PandaPushEnv(1)

# Get the height of the table
h = env.get_table_height()

# Initialization - positions of the legos
objectid1Pos = np.array([.4, .1, h])  # green
objectid2Pos = np.array([.5, -.15, h])  # red
objectid3Pos = np.array([.6, .15, h])  # blue

# Initial orientation
objectid1Or = p.getQuaternionFromEuler([0, 0, 0])
objectid2Or = p.getQuaternionFromEuler([0, 0, math.pi / 4])
objectid3Or = p.getQuaternionFromEuler([0, 0, math.pi / 4])

# Store the positions in POS
POS = [objectid1Pos, objectid2Pos, objectid3Pos]
# Store the orientations in OR
OR = [objectid1Or, objectid2Or, objectid3Or]

# Load robot and object in the chosen positions
env.reset(objectid1Pos, objectid2Pos, objectid3Pos, objectid1Or, objectid2Or, objectid3Or, folder_path="./../../urdf")

### Capture the pose of the objects
observation = env._get_state()

# Update actual positions of the objects
POS, OR, ANGLES = update_pose(observation)

# Select object to push
ob_id = 0

# generate potential map
n_discretization = 50
values_2d, gx, gy, rx, ry, step = generate_map(box_length=0.032, x_green=POS[0][0], y_green=POS[0][1], x_red=POS[1][0], y_red=POS[1][1],
                                               x_blue=POS[2][0], y_blue=POS[2][1], psi_green=ANGLES[0], psi_red=ANGLES[1], psi_blue=ANGLES[2],
                                               minX=minX, maxX=maxX, minY=minY, maxY=maxY,
                                               N=n_discretization, show_plot=True)

# The threshold represent the minimal value in the potential map that we want the final point to have
threshold = np.amax(values_2d) - 0.15

# Value associated to the points in the map to avoid
default_value = -100

# run the BFS algorithm for the given initial position of the object to push
P = best_first_search(x_start_pos=POS[ob_id][0], y_start_pos=POS[ob_id][1], values_2d=values_2d, gx=gx, gy=gy, rx=rx, ry=ry, step=step,
                      threshold=threshold, default_value=default_value, verbose=False, show_animation=False, show_final_result=True)

# if the path is not empty
if not P == None:
    path = []
    for elem in P:
        path.append(np.array([elem[0], elem[1], h]))

    # Plot debug lines to visualize the path
    env.plot_path(path)

    # Move the object ob_id along the path
    env.track(path, ob_id)

    # Prints statistics
    env.print_err()

    # Deletes debug lines
    env.deletes_lines()

# Second path


### Capture the pose of the objects
observation = env._get_state()

# Update actual positions of the objects
POS, OR, ANGLES = update_pose(observation)

# Select object to push
ob_id = 0

# generate potential map
n_discretization = 50
values_2d, gx, gy, rx, ry, step = generate_map(box_length=0.032, x_green=POS[0][0], y_green=POS[0][1], x_red=POS[1][0], y_red=POS[1][1],
                                               x_blue=POS[2][0], y_blue=POS[2][1], psi_green=ANGLES[0], psi_red=ANGLES[1], psi_blue=ANGLES[2],
                                               minX=minX, maxX=maxX, minY=minY, maxY=maxY,
                                               N=n_discretization, show_plot=True)

# The threshold represent the minimal value in the potential map that we want the final point to have
threshold = np.amax(values_2d) - 0.05

# Value associated to the points in the map to avoid
default_value = -100

# run the BFS algorithm for the given initial position of the object to push
P = best_first_search(x_start_pos=POS[ob_id][0], y_start_pos=POS[ob_id][1], values_2d=values_2d, gx=gx, gy=gy, rx=rx, ry=ry, step=step,
                      threshold=threshold, default_value=default_value, verbose=False, show_animation=False, show_final_result=True)

# if the path is not empty
if not P == None:
    path = []
    for elem in P:
        path.append(np.array([elem[0], elem[1], h]))

    # Plot debug lines to visualize the path
    env.plot_path(path)

    # Move the object ob_id along the path
    env.track(path, ob_id)

    # Prints statistics
    env.print_err()

    # Deletes debug lines
    env.deletes_lines()

# Press a key to exit
input("press")

# delete the envronment once the smulation is over
env.close()
