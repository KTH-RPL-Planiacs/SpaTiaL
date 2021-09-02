import math
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

#########
# Parameters
#########

id = -1  # id debug line
min_ray = 0.38
max_ray = 0.7


##########


#########
# Functions
#########
def update_ANGLES(POS, OR):
    ANGLES = []

    for ob_id in range(9):
        if ob_id == 1 or ob_id == 3 or ob_id == 4 or ob_id == 6:
            final_angle = compute_angle(POS[ob_id], OR, ob_id)

        # otherwise it is not necessary
        else:
            final_angle = 0

        ANGLES.append(final_angle)

    return ANGLES


def update_pose(observation):
    objectid1Pos = observation[7:10]
    objectid2Pos = observation[14:17]
    objectid3Pos = observation[21:24]

    objectid4Pos = observation[28:31]
    objectid5Pos = observation[35:38]
    objectid6Pos = observation[42:45]

    objectid7Pos = observation[49:52]
    objectid8Pos = observation[56:59]
    objectid9Pos = observation[63:66]

    POS = [objectid1Pos, objectid2Pos, objectid3Pos, objectid4Pos, objectid5Pos, objectid6Pos, objectid7Pos, objectid8Pos, objectid9Pos]

    objectid1Or = observation[10:14]
    objectid2Or = observation[17:21]
    objectid3Or = observation[24:28]

    objectid4Or = observation[31:35]
    objectid5Or = observation[38:42]
    objectid6Or = observation[45:49]

    objectid7Or = observation[52:56]
    objectid8Or = observation[59:63]
    objectid9Or = observation[66:71]

    OR = [objectid1Or, objectid2Or, objectid3Or, objectid4Or, objectid5Or, objectid6Or, objectid7Or, objectid8Or, objectid9Or]

    return POS, OR


def compute_angle(pos_ob, OR, ob_id):
    orientation_matrix = np.array(p.getMatrixFromQuaternion(OR[ob_id])).reshape(3, 3)
    axis = np.matmul(orientation_matrix, np.array([0, 1, 0]))
    unit_vector_1 = axis / np.linalg.norm(axis)
    unit_vector_2 = np.array([1, 0, 0])
    dot_product = np.dot(unit_vector_1, unit_vector_2)

    or_ob = np.arccos(dot_product)

    or_ob = or_ob * np.sign(unit_vector_1[1])

    final_angle = or_ob
    # Back to (-pi,pi) range
    signum = np.sign(final_angle)
    if abs(final_angle) > math.pi:
        final_angle = signum * (abs(final_angle) % (math.pi))

    # if beyond the joint limit, consider an equivalent angle
    if final_angle > 3 * math.pi / 4:
        final_angle = final_angle - math.pi
    elif final_angle < -3 * math.pi / 4:
        final_angle = final_angle + math.pi

    lineWidth = 1.8
    point2 = pos_ob + unit_vector_1 / 2

    # global id
    # # p.removeAllUserDebugItems()
    # if not id == -1:
    #     p.removeUserDebugItem(id)
    # id = p.addUserDebugLine((pos_ob[0],pos_ob[1], h), (point2[0],point2[1],h), [1,0,0], lineWidth)

    return final_angle


################


#########
# PandaGraspEnv class definition
#########

class PandaGraspEnv:
    def __init__(self, gui=0):
        """
        __init__: initializes the environment and set the parameters
        """

        self.step_counter = 0

        if gui:
            p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
        else:
            p.connect(p.DIRECT)

        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=90, cameraPitch=-45,
                                     cameraTargetPosition=[0, .35, 0.6])  # the initial view of the environment
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        p.setRealTimeSimulation(0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        ###########
        # Parameters

        # threshold to check if an object has fallen
        self.threshold_orient_fall = 0.07

        # tolerance with which the object is considered in the target position
        self.tol = 0.05

        # Region reachable by the robot
        self.min_ray = 0.38
        self.max_ray = 0.7
        self.max_angle = 3 * math.pi / 4
        self.min_angle = -3 * math.pi / 4

        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01]

        self.pandaNumDofs = 7

        # rest position of the robot
        self.rp = [0, -0.215, 0, -2.57, 0, 2.356, 0, 0.008, 0.008]

        # self.logId = -1

        self.tableheight = .94
        self.finger_target_open = 0.04
        self.finger_target_close = 0.005

        ## Parameters for tests
        # number of times an object has fallen
        self.counter_fallen = 0
        # number of tiimes an object is out of the reachable region
        self.counter_out = 0
        # number of fails in reaching the target position
        self.action_fail = 0
        # number of actions performed
        self.counter_actions = 0

        ######

    def restore_joint(self):
        """
        restore_joint: restores initial joints position
        """
        for i in range(1, self.pandaNumDofs):
            p.setJointMotorControl2(bodyIndex=self.pandaUid,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.rp[i],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)

        p.stepSimulation()
        time.sleep(self.timeStep)

    def motor_control(self, pos, orn):
        """
        motor_control: computes inverse kinematics and performes motor control
        """

        current_joint = np.zeros(self.pandaNumDofs)
        for i in range(self.pandaNumDofs):
            current_joint[i] = p.getJointState(self.pandaUid, i)[0]

        # jointPoses = p.calculateInverseKinematics(self.pandaUid, self.pandaEndEffectorIndex, pos, lowerLimits = self.ll, upperLimits = self.ul, jointRanges = self.jr, restPoses = self.rp)
        jointPoses = p.calculateInverseKinematics(self.pandaUid, self.pandaEndEffectorIndex, pos, orn, lowerLimits=self.ll, upperLimits=self.ul,
                                                  jointRanges=self.jr, restPoses=current_joint, jointDamping=self.jd)

        for i in range(self.pandaNumDofs):
            p.setJointMotorControl2(bodyIndex=self.pandaUid,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
            p.stepSimulation()
            time.sleep(self.timeStep)

    # Move to goal pose
    def move(self, pos_goal, psi):
        """
        move: calls the functions to move the end-effector in pos_goal with desired
        orientation
        """

        threshold_pos = 0.002
        threshold_or = 0.001
        cnt = 0

        Maxiters = 100

        while True:

            or_ee = p.getQuaternionFromEuler([-math.pi, 0., psi])

            self.motor_control(pos_goal, or_ee)

            self._get_state()

            pos_cur = self.observation[0:3]
            de = pos_goal - pos_cur
            err_pos = np.linalg.norm(de)

            or_cur = p.getEulerFromQuaternion(p.getLinkState(self.pandaUid, 11)[1])[2]

            # Back to (-pi,pi) range
            signum = np.sign(or_cur)
            if abs(or_cur) > math.pi:
                or_cur = signum * (abs(or_cur) % (math.pi))

            do = or_cur - (psi)
            err_or = np.linalg.norm(do)

            cnt += 1

            if (err_pos < threshold_pos and err_or < threshold_or) or cnt > Maxiters:
                break

        return self.observation[0:3], p.getEulerFromQuaternion(self.observation[3:7])[2]

    def move_fingers(self, finger_target):
        """
        move_fingers: changes the fingers positions of the gripper
        """

        for i in range(150):
            for i in [9, 10]:
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, finger_target,
                                        force=50,
                                        positionGain=0.03,
                                        velocityGain=1)

            p.stepSimulation()
            time.sleep(self.timeStep)

    def rotate(self, rot):
        """
        rotate: rotate aroud the base joint
        """

        threshold_or = 0.00001
        cnt = 0
        Maxiters = 200

        while True:

            current = p.getJointState(self.pandaUid, 0)[0]
            dj = rot - current
            err_joint = np.linalg.norm(dj)

            p.setJointMotorControl2(bodyIndex=self.pandaUid,
                                    jointIndex=0,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=rot,
                                    targetVelocity=0,
                                    force=100,
                                    positionGain=0.003,
                                    velocityGain=0.2)
            p.stepSimulation()
            time.sleep(self.timeStep)

            cnt += 1

            if err_joint < threshold_or or cnt > Maxiters:
                break

    def step(self, action, ob_id):
        """
        step: performes the robot action from picking the object to placing it

        ACTION:
         - position to pick [array of 3 elements]
         - position to place [array of 3 elements]
         - end effector orientation (yaw) [scalar]
         - angle base for picking [scalar]
         - angle base for placing [scalar]

        """

        pos_goal = action[0:3]
        pos_goal2 = action[3:6]
        self.psi = action[6]
        rot = action[7]
        rot2 = action[8]

        # check if the rotation angle is compatible with the joint limits, otherwise saturates it
        if rot2 < self.ll[0]:
            rot2 = self.ll[0]
        elif rot2 > self.ul[0]:
            rot2 = self.ul[0]

        # Parameters grasping according to the object
        if ob_id == 0:  # bottle
            z_up = self.tableheight + 0.35
            z_down = self.tableheight + 0.15

            self.finger_target_open = 0.08
            self.finger_target_close = 0.001

        elif ob_id == 1:  # banana
            z_up = self.tableheight + 0.35
            z_down = self.tableheight + 0.013

            self.finger_target_open = 0.04
            self.finger_target_close = 0.001

        elif ob_id == 2:  # mug
            z_up = self.tableheight + 0.35
            z_down = self.tableheight + 0.03

            self.finger_target_open = 0.08
            self.finger_target_close = 0.001

        elif ob_id == 3:  # gelatin
            z_up = self.tableheight + 0.35
            z_down = self.tableheight + 0.04

            self.finger_target_open = 0.08
            self.finger_target_close = 0.001


        elif ob_id == 4:  # sugar
            z_up = self.tableheight + 0.35
            z_down = self.tableheight + 0.15

            self.finger_target_open = 0.08
            self.finger_target_close = 0.001

        elif ob_id == 5:  # master chef can
            z_up = self.tableheight + 0.35
            z_down = self.tableheight + 0.05

            self.finger_target_open = 0.08
            self.finger_target_close = 0.001

        elif ob_id == 6:  # cracker
            z_up = self.tableheight + 0.35
            z_down = self.tableheight + 0.15

            self.finger_target_open = 0.08
            self.finger_target_close = 0.001

        elif ob_id == 7:  # kanelbulle
            z_up = self.tableheight + 0.35
            z_down = self.tableheight + 0.01

            self.finger_target_open = 0.08
            self.finger_target_close = 0.001

        print('Manipulator action execution')
        print('reach safe position, change end-effector orientation and open fingers')

        # Rotate joint 0
        print('rotate')
        self.rotate(rot)

        pos_goal[2] = z_up
        # fingers control function
        finger_target = self.finger_target_open
        self.move_fingers(finger_target)

        pos_cur, or_cur = self.move(pos_goal, self.psi)

        print('reach position from which it is possible to grasp the object')
        pos_goal[2] = z_down
        pos_cur, or_cur = self.move(pos_goal, self.psi)

        print('grasp the object')
        finger_target = self.finger_target_close
        self.move_fingers(finger_target)

        # going back
        print('going back')
        pos_goal[2] = z_up

        pos_cur, or_cur = self.move(pos_goal, self.psi)

        # Rotate
        print('rotate')

        # if the robot is rotating from one side of the table to the other one
        # then the joint angle of the first joint changes sign. In this case reach
        # the intermediate point angle = 0, before reaching rot2 (to avoid dropping
        # the object)
        if rot * rot2 < 0:
            self.rotate(0)

        self.rotate(rot2)

        print('reach position to release the object')
        pos_goal2[2] = z_up
        pos_cur, or_cur = self.move(pos_goal2, self.psi)

        pos_goal2[2] = z_down + .02
        pos_cur, or_cur = self.move(pos_goal2, self.psi)

        finger_target = self.finger_target_open
        self.move_fingers(finger_target)

        # reset orientation
        print('reset orientation end effector and go up')
        pos_goal2[2] = z_up

        pos_cur, or_cur = self.move(pos_goal2, self.psi * 0)

        self.restore_joint()

        return np.array(self.observation).astype(np.float32)

    def grasp(self, ob_id, pos_ob2, verbose=False):
        """
        grasp: moves the object identified by ob_id in position pos_ob2
        """

        # get the current poses
        observation = self._get_state()
        POS, OR = update_pose(observation)

        # position of selected object
        pos_ob = POS[ob_id]

        # temporary fix - is this proper?
        ANGLES = update_ANGLES(POS, OR)

        # rotation to execute to position the gripper in front of the object to pick
        rot_angle = math.atan2(pos_ob[1], pos_ob[0])
        # rotation to execute to position the gripper in front of the object to place
        rot_angle2 = math.atan2(pos_ob2[1], pos_ob2[0])
        # generate the action
        a = np.concatenate((pos_ob, pos_ob2, ANGLES[ob_id], rot_angle, rot_angle2), axis=None)

        # perform the action
        observation = self.step(a, ob_id)

        # Update position and orientation of the objects
        POS, OR = update_pose(observation)

        # # check if one of the object fell or went out of the rechaeble region
        fallen, out = self.check(POS, OR, self.OR_init)

        # get observation again in case an object is fallen or was out of reachable region
        if fallen > 0 or out > 0:
            observation = self._get_state()
            POS, OR = update_pose(observation)

        # check if the object has been moved in the target position
        self.check_success(pos=POS[ob_id][0:2], target=pos_ob2[0:2])

        self.counter_actions += 1

        if verbose:
            self.print_statistics()

        ANGLES = update_ANGLES(POS, OR)

        return POS, OR, ANGLES

    def reset(self, P, O, folder_path):
        """
        reset: loads the robot, the objects and the environment and set the initial poses
        """

        self.step_counter = 0
        self.prevPose = [0, 0, 0]
        self.prevPose1 = [0, 0, 0]
        self.hasPrevPose = 0
        self.trailDuration = 15

        self.OR_init = O

        p.resetSimulation()  # reset the PyBullet environment
        p.setPhysicsEngineParameter(numSolverIterations=150)
        fps = 120.
        self.timeStep = 1. / fps
        p.setTimeStep(self.timeStep)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # disable the rendering

        p.setGravity(0, 0, -9.81)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Load objects
        urdfRootPath = pybullet_data.getDataPath()
        # Plane
        self.planeUid = p.loadURDF(folder_path + "/floor/floor.urdf", useFixedBase=True)
        # self.planeUid = p.loadURDF("./urdf/black_floor.urdf", basePosition=[0,-.5,0.0])

        # kitchen
        # kitchen_path = './urdf/models/kitchen_description/urdf/kitchen_part_right_gen_convex.urdf'
        kitchen_path = folder_path + "/kitchen/kitchen_description/urdf/kitchen_part_right_gen_convex.urdf"
        self.kitchen = p.loadURDF(kitchen_path, [-2, 0., 1.477], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

        wall = folder_path + "/kitchen_walls/model_normalized.urdf"
        self.wall = p.loadURDF(wall, [-2, -3.5, 1.477], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        p.changeVisualShape(self.wall, -1, rgbaColor=[0.7, 0.7, 0.7, 1])
        self.wall2 = p.loadURDF(wall, [-2, 3.5, 1.477], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        p.changeVisualShape(self.wall2, -1, rgbaColor=[0.7, 0.7, 0.7, 1])

        ## Objects ##
        # bottle
        self.objectid1 = p.loadURDF(folder_path + "/bottle/bottle.urdf", P[0], O[0], globalScaling=0.04)

        # banana
        banana_path = folder_path + "/011_banana/model_normalized.urdf"
        self.objectid2 = p.loadURDF(banana_path, P[1], O[1])

        # mug1
        mug_path = folder_path + "/025_mug/model_normalized.urdf"
        self.objectid3 = p.loadURDF(mug_path, P[2], O[2], globalScaling=.6)

        # gelatin
        gelatin_path = folder_path + "/009_gelatin_box/model_normalized.urdf"
        self.objectid4 = p.loadURDF(gelatin_path, P[3], O[3])

        # sugar box
        sugar_path = folder_path + "/004_sugar_box/model_normalized.urdf"
        self.objectid5 = p.loadURDF(sugar_path, P[4], O[4])

        can_path = folder_path + "/002_master_chef_can/model_normalized.urdf"
        self.objectid6 = p.loadURDF(can_path, P[5], O[5], globalScaling=.6)

        cracker_path = folder_path + "/003_cracker_box/model_normalized.urdf"
        self.objectid7 = p.loadURDF(cracker_path, P[6], O[6], globalScaling=.8)

        Kanelbulle_path = folder_path + "/kanel/kanel.urdf"
        self.objectid8 = p.loadURDF(Kanelbulle_path, P[7], O[7], globalScaling=0.7)
        # p.changeVisualShape(self.objectid8 , -1, rgbaColor=[150/256,75/256,0,1])
        texture_path = folder_path + "/kanel/kanel.jpg"
        textureId = p.loadTexture(texture_path)
        p.changeVisualShape(self.objectid8, -1, textureUniqueId=textureId)

        # plate
        plate_path = folder_path + "/plate/plate.urdf"
        self.objectid9 = p.loadURDF(plate_path, P[8], O[8])
        p.changeVisualShape(self.objectid9, -1, rgbaColor=[1, 1, 1, 1])

        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.tableUid = p.loadURDF("table/table.urdf", [0., 0, 0], p.getQuaternionFromEuler([0, 0, math.pi / 2]), globalScaling=1.5)

        # Franka Panda robot
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), [0, 0, self.tableheight], useFixedBase=True)
        self.pandaEndEffectorIndex = 11

        # create a constraint to keep the fingers centered
        c = p.createConstraint(self.pandaUid,
                               9,
                               self.pandaUid,
                               10,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=800)

        # Initial position robot
        for i in range(7):
            p.resetJointState(self.pandaUid, i, self.rp[i])
        p.resetJointState(self.pandaUid, 9, self.rp[7])
        p.resetJointState(self.pandaUid, 10, self.rp[8])

        # Enable the rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Get current state
        self._get_state()

        # Plot of reachable region
        lineWidth = 1.8
        color = [0, 0, 0]

        # theta = np.linspace(0, 2*math.pi, 100)
        #
        # r = self.min_ray
        # x1 = r*np.cos(theta)
        # x2 = r*np.sin(theta)
        #
        # for k in range(len(x1)-1):
        #     p.addUserDebugLine((x1[k],x2[k], self.tableheight), (x1[k+1],x2[k+1],self.tableheight), color, lineWidth)
        #
        # r = self.max_ray
        # x1 = r*np.cos(theta)
        # x2 = r*np.sin(theta)
        #
        # for k in range(len(x1)-1):
        #     p.addUserDebugLine((x1[k],x2[k], self.tableheight), (x1[k+1],x2[k+1],self.tableheight), color, lineWidth)

        # p.addUserDebugLine((min,min2, self.tableheight), (min,max2,self.tableheight), color, lineWidth)
        # p.addUserDebugLine((min,min2, self.tableheight), (max,min2,self.tableheight), color, lineWidth)

        # p.addUserDebugLine((max,max2, self.tableheight), (min,max2,self.tableheight), color, lineWidth)
        # p.addUserDebugLine((max,max2, self.tableheight), (max,min2,self.tableheight), color, lineWidth)

        # if self.logId==-1:
        #     self.logId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video.mp4")

        self._update_constraints()

    def _update_constraints(self):
        """
        _update_constraints: updates internal constraints on joints and tools
        """

        self.ll = [p.getJointInfo(self.pandaUid, joint)[8] for joint in range(self.pandaNumDofs)]
        self.ul = [p.getJointInfo(self.pandaUid, joint)[9] for joint in range(self.pandaNumDofs)]

        self.jr = [0] * self.pandaNumDofs
        for k in range(self.pandaNumDofs):
            self.jr[k] = abs(self.ll[k] - self.ul[k])

    def get_table_height(self):
        """
        Tableheight: returns the height of the table
        """

        return self.tableheight

    def plot_traj(self, pos):
        """
        plot_traj: plots the trajectory tracked by end-effector
        """

        # plot trajectory
        ls = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)
        if (self.hasPrevPose):
            # p.addUserDebugLine(self.prevPose, pos, [0, 0, 0.3], 1, self.trailDuration)
            p.addUserDebugLine(self.prevPose1, ls[4], [1, 0, 0], 1, self.trailDuration)
        self.prevPose = pos
        self.prevPose1 = ls[4]
        self.hasPrevPose = 1

    def _get_state(self):
        """
        _get_state: returns the current state
        """

        state_robot_pos = p.getLinkState(self.pandaUid, 11)[0]  # 3D pos end-effector
        state_robot_or = p.getLinkState(self.pandaUid, 11)[1]  # 4D orientation end-effector
        state_object_pos1, state_object_or1 = p.getBasePositionAndOrientation(self.objectid1)
        state_object_pos2, state_object_or2 = p.getBasePositionAndOrientation(self.objectid2)
        state_object_pos3, state_object_or3 = p.getBasePositionAndOrientation(self.objectid3)

        state_object_pos4, state_object_or4 = p.getBasePositionAndOrientation(self.objectid4)
        state_object_pos5, state_object_or5 = p.getBasePositionAndOrientation(self.objectid5)
        state_object_pos6, state_object_or6 = p.getBasePositionAndOrientation(self.objectid6)

        state_object_pos7, state_object_or7 = p.getBasePositionAndOrientation(self.objectid7)
        state_object_pos8, state_object_or8 = p.getBasePositionAndOrientation(self.objectid8)
        state_object_pos9, state_object_or9 = p.getBasePositionAndOrientation(self.objectid9)

        self.observation = np.concatenate((state_robot_pos, state_robot_or, state_object_pos1, state_object_or1, state_object_pos2, state_object_or2,
                                           state_object_pos3, state_object_or3, state_object_pos4, state_object_or4, state_object_pos5,
                                           state_object_or5, state_object_pos6, state_object_or6, state_object_pos7, state_object_or7,
                                           state_object_pos8, state_object_or8, state_object_pos9, state_object_or9), axis=0)

        return self.observation

    def restore_pose(self, pos, orient, ob_id):
        """
        restore_pose: restores the position and orientation of object ob_id
        """

        ID = [self.objectid1, self.objectid2, self.objectid3, self.objectid4, self.objectid5, self.objectid6, self.objectid7, self.objectid8,
              self.objectid9]
        p.resetBasePositionAndOrientation(ID[ob_id], pos, orient)

        # self.changeObjPoseCamera(objectid1Pos,orientation,objectid2Pos,orientation,objectid3Pos,orientation)
        p.stepSimulation()
        # time.sleep(self.timeStep)

    def check(self, POS, OR, OR_init):
        """
        check: checks if one of the objects has fallen or gone out of the rechaeble region
        """
        fallen = False
        out = False

        for j in range(9):

            # Has j-object fallen?
            if abs(p.getEulerFromQuaternion(p.getDifferenceQuaternion(OR[j], OR_init[j]))[0]) > self.threshold_orient_fall or abs(
                    p.getEulerFromQuaternion(p.getDifferenceQuaternion(OR[j], OR_init[j]))[1]) > self.threshold_orient_fall:
                print("**************Object ", j, " has fallen **************")

                self.counter_fallen += 1
                fallen = True

                self.restore_pose(np.append(POS[j][0:2], self.tableheight + 0.03), OR_init[j], j)

            # Has j-object gone out of the rechaeble region?
            distance_from_origin = np.linalg.norm(POS[j][0:2])

            if distance_from_origin < self.min_ray or distance_from_origin > self.max_ray:
                print("**************Object ", j, " is out **************")

                self.counter_out += 1
                out = True

                if distance_from_origin < self.min_ray:
                    ray = self.min_ray + .02
                else:
                    ray = self.max_ray - .02

                angle = math.atan2(POS[j][1], POS[j][0])
                POS[j] = np.array([ray * math.cos(angle), ray * math.sin(angle), self.tableheight + 0.13])

                self.restore_pose(POS[j], OR_init[j], j)

        return fallen, out

    def check_success(self, pos, target):
        """
        check_success: checks if the object is in the target position
        """

        if np.linalg.norm(pos - target) > self.tol:
            print("**********Action failed**************")
            self.action_fail += 1
            self.restore_joint()

    def print_statistics(self):
        """
        print_statistics: prints success/fail scores and number of executed actions
        """

        print("Number of executed actions: ", self.counter_actions)
        print("Out counter: ", self.counter_out)
        print("Fallen counter: ", self.counter_fallen)
        print("Action fails counter: ", self.action_fail)

    def close(self):
        """
        close: deletes the environment
        """

        # p.stopStateLogging(self.logId)

        p.disconnect()
