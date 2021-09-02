"""
Pybullet Franka Emika Robot - Grasping task

This file provides an example of grasping simulation for the following formula:

formula = "(bottle dist banana >= 0.05) & (bottle dist mug >= 0.05) & (bottle above banana) & (bottle rightof banana)"

id                    name
0                   "bottle",
1                   "banana",
2                   "mug"

Only moving the bottle or the banana you can satisfy the specification, so chosing
id = 2 you get the message "Moving this object does not change the level of satisfaction!"
Chosing id = 0,1 instead lets you find a configuration of items that satisfies the specification
"""

import sys

from lib import *

sys.path.append("./../../scripts")
from gradient_grasp import generate_map

###########
# Simulation


# Create the environment
env = PandaGraspEnv(1)

# Initialization - position
h = env.get_table_height() + 0.08

objectid1Pos = np.array([-0.15, 0.5, h])  # bottle
objectid2Pos = np.array([0.45, 0.23, h])  # banana
objectid3Pos = np.array([0.45, -0.2, h])  # mug red
objectid4Pos = np.array([0., 0.5, h - 0.05])  # gelatin
objectid5Pos = np.array([0.3, 0.4, h])  # sugar_box
objectid6Pos = np.array([0.3, -0.4, h])  # master chef can
objectid7Pos = np.array([0., -0.45, h])  # cracker_box
objectid8Pos = np.array([0.2, 0.55, h - .01])  # kanelbulle
objectid9Pos = np.array([0.4, 0., h])  # plate

# Initialization - orientation
objectid1Or_init = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
objectid2Or_init = p.getQuaternionFromEuler([0, 0, 0])
objectid3Or_init = p.getQuaternionFromEuler([0, 0, 0])
objectid4Or_init = p.getQuaternionFromEuler([math.pi / 2, +math.pi / 2 + math.pi / 14, math.pi / 2])
objectid5Or_init = p.getQuaternionFromEuler([0, 0, math.pi / 2])
objectid6Or_init = p.getQuaternionFromEuler([0, 0, 0])
objectid7Or_init = p.getQuaternionFromEuler([0, 0, math.pi / 2])
objectid8Or_init = p.getQuaternionFromEuler([0, 0, 0])
objectid9Or_init = p.getQuaternionFromEuler([0, 0, 0])

# Store initial orientation in OR_init
OR_init = [objectid1Or_init, objectid2Or_init, objectid3Or_init, objectid4Or_init, objectid5Or_init, objectid6Or_init, objectid7Or_init,
           objectid8Or_init, objectid9Or_init]

# Store position in POS
POS = [objectid1Pos, objectid2Pos, objectid3Pos, objectid4Pos, objectid5Pos, objectid6Pos, objectid7Pos, objectid8Pos, objectid9Pos]

# Current orientation
OR = OR_init

# Angle between the x-axis and the object axis
ANGLES = update_ANGLES(POS, OR)

# Load robot and object in the chosen pose
env.reset(POS, OR_init, folder_path="./../../urdf")

# selected object to grasp
ob_id = 1  # from 0 to 7

# generate map and select the place position
n_discretization = 20
pos_ob2 = generate_map(ob_id, POS, ANGLES, minX=-max_ray, maxX=max_ray, minY=-max_ray, maxY=max_ray, minor_circle=min_ray, N=n_discretization,
                       show_plot=True)

# move the object if there is a position where it can be place to improve the
# level of satisfaction of the specification
if not (pos_ob2 == POS[ob_id]).all():
    # Move the object identified by ob_id in position pos_ob2
    POS, OR, ANGLES = env.grasp(ob_id, pos_ob2)

# Press a key to exit
input('press...')

# delete the envronment once the smulation is over
env.close()
