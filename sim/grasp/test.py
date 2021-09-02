"""
Pybullet Franka Emika Robot - Grasping task

Test pick and place of the bottle in the area behind the Robot (positions randomly generated)
Results obtained for one of the executed simulations:

    Number of executed actions:  100
    Out counter:  1
    Fallen counter:  0
    Action fails counter:  0

"""

from lib import *

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
ob_id = 0  # from 0 to 7

while True:
    # target position generation
    pos_ob2 = np.array([-0.4, np.random.uniform(-0.05, 0.05), h])  # bottle

    # Move the object identified by ob_id in position pos_ob2
    POS, OR, ANGLES = env.grasp(ob_id, pos_ob2, verbose=True)

input('press something...')

# delete the envronment once the smulation is over
env.close()
