import math
import os
import time

import numpy as np
import pybullet as p
import pybullet_data


#########
# Functions
#########

def update_pose(observation):
    objectid1Pos = observation[7:10]
    objectid2Pos = observation[14:17]
    objectid3Pos = observation[21:24]
    POS = [objectid1Pos, objectid2Pos, objectid3Pos]
    objectid1Or = observation[10:14]
    objectid2Or = observation[17:21]
    objectid3Or = observation[24:28]
    OR = [objectid1Or, objectid2Or, objectid3Or]

    _, _, psi1 = p.getEulerFromQuaternion(objectid1Or)
    _, _, psi2 = p.getEulerFromQuaternion(objectid2Or)
    _, _, psi3 = p.getEulerFromQuaternion(objectid3Or)

    ANGLES = [psi1, psi2, psi3]

    return POS, OR, ANGLES


#########


#########
# Parameters
#########

# Region reachable by the robot
minX = 0.28
maxX = 0.73
minY = -0.225
maxY = 0.225


####


#########
# PandaPushEnv class definition
#########


class PandaPushEnv:
    def __init__(self, gui=0):
        """
        __init__: initializes the environment and set the parameters
        """

        self.step_counter = 0

        if gui:
            p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
        else:
            p.connect(p.DIRECT)
        # p.connect(p.GUI,options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')

        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-30,
                                     cameraTargetPosition=[0.55, -0.35, 0.9])  # the initial view of the environment
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        p.setRealTimeSimulation(0)
        # p.setRealTimeSimulation(1)

        self.tableheight = 0.63

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.logId = -1

        # initialize list of segments of the tracked path
        self.idsegment = []
        self.obline = -1

        self.pandaNumDofs = 7

        self.ll = [-7] * self.pandaNumDofs
        # upper limits for null space (todo: set them to proper range)
        self.ul = [7] * self.pandaNumDofs
        # joint ranges for null space (todo: set them to proper range)
        self.jr = [7] * self.pandaNumDofs

        # rest position of the robot
        self.rp = [0, -0.215, 0, -2.57, 0, 2.356, 0, 0.001, 0.001]

    def motor_control(self, pos, orn):
        """
        motor_control: computes inverse kinematics and performs motor control
        """

        jointPoses = p.calculateInverseKinematics(self.pandaUid, self.pandaEndEffectorIndex, pos, orn, self.ll, self.ul, self.jr, self.rp)

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

        # self.plot_traj(pos)

    def move_segment(self, pos_goal, psi, ob_id):
        """
        motor_control: moves end effector to pos_goal along the segment that link the current
        position to pos_goal, keeping psi-orientation
        """

        threshold_pos = 0.0001
        threshold_or = 0.001

        # tolerance object position
        tol_dist_line = 0.005

        cnt = 0

        # initial position end effector
        pos_init = self.observation[0:3]

        maxIters = 300

        while True:

            # desired end-effector orientation
            or_ee = p.getQuaternionFromEuler([-math.pi, 0., psi])

            # call computeInverseKinematics and jointmotorcontrol functions
            self.motor_control(pos_goal, or_ee)

            # update self.observation
            self._get_state()

            # Update actual positions of the objects
            objectid1Pos = self.observation[7:10]
            objectid2Pos = self.observation[14:17]
            objectid3Pos = self.observation[21:24]
            POS = [objectid1Pos, objectid2Pos, objectid3Pos]

            pos_ob = POS[ob_id]

            # if self.obline:
            #     p.removeUserDebugItem(self.obline)
            # self.obline = p.addUserDebugLine((pos_ob[0],pos_ob[1], self.tableheight), (pos_ob[0],pos_ob[1],self.tableheight+0.05), color, lineWidth)

            # distance to the target
            dist_target = np.linalg.norm(pos_goal - pos_ob)

            if dist_target < self.tol:
                break

            # compute the position error
            pos_cur = self.observation[0:3]
            de = pos_goal - pos_cur
            err_pos = np.linalg.norm(de)

            # compute orientation error (only last angle)
            or_cur = p.getEulerFromQuaternion(p.getLinkState(self.pandaUid, 11)[1])[2]

            # Back to (-pi,pi) range
            signum = np.sign(or_cur)
            if abs(or_cur) > math.pi:
                or_cur = signum * (abs(or_cur) % (math.pi))

            do = or_cur - (psi)
            err_or = np.linalg.norm(do)

            cnt += 1

            # distance object to the segment
            d = np.linalg.norm(np.cross(pos_init[0:2] - pos_goal[0:2], pos_goal[0:2] - pos_ob[0:2])) / np.linalg.norm(pos_init[0:2] - pos_goal[0:2])

            if d > tol_dist_line:
                print("Too far from the segment ", d, pos_goal)
                push_angle = math.atan2(pos_goal[1] - pos_ob[1], pos_goal[0] - pos_ob[0])

                a = np.concatenate((push_angle, pos_ob, pos_goal), axis=None)
                self.step(a, ob_id)

                # # input("press")
                # # go back to the line
                # unit_vector_1 = (pos_ob-pos_init)/np.linalg.norm(pos_ob-pos_init)
                # unit_vector_2 = (pos_goal-pos_init)/ np.linalg.norm(pos_goal-pos_init)
                # cosalpha = np.dot(unit_vector_1, unit_vector_2)
                #
                # n = unit_vector_2*cosalpha*np.linalg.norm(pos_ob-pos_init)
                # vec_d = (pos_ob - pos_init) - n
                # pos_goal2 = pos_ob - vec_d
                # push_angle = math.atan2(pos_goal2[1]-pos_ob[1], pos_goal2[0]-pos_ob[0])
                #
                # a = np.concatenate((push_angle,pos_ob,pos_goal2),axis=None)
                # self.step(a,ob_id)

            # exit if the errors are smaller than the tresholds or exceeded the maximum number of iterations
            if (err_pos < threshold_pos and err_or < threshold_or) or cnt > maxIters:
                break

        # print("position ",or_cur,"/",psi)

        return self.observation[0:3], p.getEulerFromQuaternion(self.observation[3:7])[2]

    def move(self, pos_goal, psi):
        """
        move: calls the functions to move the end-effector in pos_goal with desired
        orientation
        """

        threshold_pos = 0.002
        threshold_or = 0.001
        cnt = 0

        while True:

            # desired end-effector orientation
            or_ee = p.getQuaternionFromEuler([-math.pi, 0., psi])

            # call computeInverseKinematics and jointmotorcontrol functions
            self.motor_control(pos_goal, or_ee)

            # update self.observation
            self._get_state()

            # compute the position error
            pos_cur = self.observation[0:3]
            de = pos_goal - pos_cur
            err_pos = np.linalg.norm(de)

            # compute orientation error (only last angle)
            or_cur = p.getEulerFromQuaternion(p.getLinkState(self.pandaUid, 11)[1])[2]

            # Back to (-pi,pi) range
            signum = np.sign(or_cur)
            if abs(or_cur) > math.pi:
                or_cur = signum * (abs(or_cur) % (math.pi))

            do = or_cur - (psi)
            err_or = np.linalg.norm(do)

            cnt += 1

            # exit if the errors are smaller than the tresholds or exceeded the maximum number of iterations
            if (err_pos < threshold_pos and err_or < threshold_or) or cnt > 250:
                break

        # print("position ",or_cur,"/",psi)

        return self.observation[0:3], p.getEulerFromQuaternion(self.observation[3:7])[2]

    # execute action
    def step(self, action, ob_id):
        """
        step: performs the robot actions to push the object from the initial to
        the final position
        Action
            a[0] = end-effector orientation
            a[1:4] = object initial position
            a[4:7] = object final position

        Observation
            - end effector 3Dposition
            - end effector orientation quaternion
            - object1 3Dposition
            - object1 orientation quaternion
            - object2 3Dposition
            - object2 orientation quaternion
            - object3 3Dposition
            - object3 orientation quaternion
        """

        print('Manipulator action execution')
        print('Reach safe position and change end-effector orientation')

        # end effector orientation
        push_angle = action[0]

        # bound angle
        if push_angle > math.pi / 2:
            c = -1
            push_angle = push_angle - math.pi
        elif push_angle < -math.pi / 2:
            c = -1
            push_angle = push_angle + math.pi
        else:
            c = 1

        # approx ray of the object to push
        ray = 0.045

        # height when not pushing
        z_up = self.tableheight + 0.2

        # end-effector initial position = object initial position (center of mass of the object) -/+ offset (the signum depends on the pushing direction)
        pos_goal = action[1:4] - np.array([ray * math.cos(push_angle), ray * math.sin(push_angle), 0.]) * c
        pos_goal[2] = z_up

        # reach initial position (but at safe height)
        pos_cur, or_cur = self.move(pos_goal, push_angle)

        # go down - reach position from which start pushing
        print('reach position from which start pushing')
        z_down = self.tableheight + 0.008
        pos_goal[2] = z_down

        pos_cur, or_cur = self.move(pos_goal, push_angle)

        # pushing
        print('pushing')
        # end-effector final position = object final position (center of mass of the object) -/+ offset (the signum depends on the pushing direction)
        pos_goal = action[4:]
        # pos_goal = action[4:] - np.array([ray*math.cos(push_angle),ray*math.sin(push_angle),0.]) * c

        z_down = self.tableheight + 0.01
        pos_goal[2] = z_down

        pos_cur, or_cur = self.move_segment(pos_goal, push_angle, ob_id)
        # pos_cur,or_cur = self.move(pos_goal,push_angle)

        #
        # # step away
        # print("step away")
        # pos_goal[2] =  z_up
        #
        # pos_cur,or_cur = self.move(pos_goal,0)

        return np.array(self.observation).astype(np.float32)

    def track(self, path, ob_id):
        """
        track: moves the object ob_id along the path
        """

        # Get objects´ position
        objectid1Pos = np.array(self.observation)[7:10]
        objectid2Pos = np.array(self.observation)[14:17]
        objectid3Pos = np.array(self.observation)[21:24]
        POS = [objectid1Pos, objectid2Pos, objectid3Pos]

        # tolerance with which the object is considered in the target position
        self.tol = 0.01

        # For each point of the path (expect the initial position) generate the action to reach it
        for k in range(1, len(path)):

            target = path[k]

            while True:
                # find the object position
                pos_ob = POS[ob_id]

                # The segment between current position and target position is
                # characterized by the distance and the angle
                push_angle = math.atan2(target[1] - pos_ob[1], target[0] - pos_ob[0])
                length = np.linalg.norm(pos_ob - target)
                print("Push angle ", push_angle)
                print("Distance ", length)

                # # Action
                # a[0] = end-effector orientation
                # a[1:4] = object initial position
                # a[4:7] = object final position
                a = np.concatenate((push_angle, pos_ob, target), axis=None)

                # Performe the action
                observation = self.step(a, ob_id)

                # Update actual positions of the objects
                objectid1Pos = observation[7:10]
                objectid2Pos = observation[14:17]
                objectid3Pos = observation[21:24]
                POS = [objectid1Pos, objectid2Pos, objectid3Pos]

                # is pushed object in the target position?
                self.Nruns += 1
                dist = np.linalg.norm(POS[ob_id][0:2] - target[0:2])
                self.sumErrors += dist
                if dist > self.Maxerr:
                    self.Maxerr = dist

                if dist < self.tol:
                    break

    def deletes_lines(self):
        """
        deletes_lines: removes all the debug lines stored in self.idsegment
        """

        for elem in self.idsegment:
            p.removeUserDebugItem(elem)

    def print_err(self):
        """
        print_err: prints statistics
        """

        print("Number of actions performed", self.Nruns)
        # print("Sum of errors between the target position and the reached position", self.sumErrors)
        print("Mean error for each action", self.sumErrors / self.Nruns)
        print("Max error found", self.Maxerr)

    def reset(self, objectid1Pos, objectid2Pos, objectid3Pos, objectid1Or, objectid2Or, objectid3Or, folder_path):
        """
        reset: loads the robot, the objects and the environment and set the initial poses
        """

        self.step_counter = 0
        self.prevPose = [0, 0, 0]
        self.prevPose1 = [0, 0, 0]
        self.hasPrevPose = 0
        self.trailDuration = 15

        # Number of actions performed
        self.Nruns = 0
        # Sum of errors between the target position and the reached position
        self.sumErrors = 0
        # Max error between the target position and the reached position
        self.Maxerr = -np.inf

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
        self.kitchen = p.loadURDF(kitchen_path, [0.3, 2, 1.477], p.getQuaternionFromEuler([0, 0, -math.pi / 2]), useFixedBase=True)

        wall = folder_path + "/kitchen_walls/model_normalized.urdf"

        self.wall = p.loadURDF(wall, [-2.8, 2, 1.477], p.getQuaternionFromEuler([0, 0, -math.pi / 2]), useFixedBase=True)
        self.wall2 = p.loadURDF(wall, [3.8, 2, 1.477], p.getQuaternionFromEuler([0, 0, -math.pi / 2]), useFixedBase=True)
        p.changeVisualShape(self.wall, -1, rgbaColor=[0.7, 0.7, 0.7, 1])
        p.changeVisualShape(self.wall2, -1, rgbaColor=[0.7, 0.7, 0.7, 1])

        self.tableUid = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
        # self.tableCameraUid = p.loadURDF("table/table.urdf",basePosition=[0,0,h]+offset_base+offset)

        gS = 1

        self.objectid1 = p.loadURDF("lego/lego.urdf", objectid1Pos, objectid1Or, globalScaling=gS)
        p.changeVisualShape(self.objectid1, -1, rgbaColor=[0, 1, 0, 1])

        self.objectid2 = p.loadURDF("lego/lego.urdf", objectid2Pos, objectid2Or, globalScaling=gS)
        p.changeVisualShape(self.objectid2, -1, rgbaColor=[1, 0, 0, 1])

        self.objectid3 = p.loadURDF("lego/lego.urdf", objectid3Pos, objectid3Or, globalScaling=gS)
        p.changeVisualShape(self.objectid3, -1, rgbaColor=[0, 0, 1, 1])

        # print(p.getDynamicsInfo(self.objectid1,-1))
        # p.changeDynamics(self.objectid1,-1, angularDamping=10)

        # Franka Panda robot
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), [0, 0, self.tableheight], useFixedBase=True)
        self.pandaEndEffectorIndex = 11

        # #create a constraint to keep the fingers centered
        c = p.createConstraint(self.pandaUid,
                               9,
                               self.pandaUid,
                               10,
                               jointType=p.JOINT_FIXED,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])

        # Initial position robot
        for i in range(7):
            p.resetJointState(self.pandaUid, i, self.rp[i])
        p.resetJointState(self.pandaUid, 9, self.rp[7])
        p.resetJointState(self.pandaUid, 10, self.rp[8])

        # Enable the rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Get current state
        self._get_state()

        # Plot the operational region
        lineWidth = 1.8
        color = [0, 0, 0]

        p.addUserDebugLine((minX, minY, self.tableheight), (minX, maxY, self.tableheight), color, lineWidth)
        p.addUserDebugLine((minX, minY, self.tableheight), (maxX, minY, self.tableheight), color, lineWidth)

        p.addUserDebugLine((maxX, maxY, self.tableheight), (minX, maxY, self.tableheight), color, lineWidth)
        p.addUserDebugLine((maxX, maxY, self.tableheight), (maxX, minY, self.tableheight), color, lineWidth)

    def plot_path(self, path):
        """
        plot_path: plots debug lines to visualize the path
        """

        color = [1, 0, 0]
        lineWidth = 1.8

        for k in range(len(path) - 1):
            point1 = path[k]
            point2 = path[k + 1]

            new_segment_id = p.addUserDebugLine((point1[0], point1[1], self.tableheight + 0.02), (point2[0], point2[1], self.tableheight + 0.02),
                                                color, lineWidth)
            self.idsegment.append(new_segment_id)

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
        # print('Link state',p.getLinkState(self.pandaUid, 11))
        # print('Euler', p.getEulerFromQuaternion(state_robot_or))

        self.observation = np.concatenate((state_robot_pos, state_robot_or, state_object_pos1, state_object_or1, state_object_pos2, state_object_or2,
                                           state_object_pos3, state_object_or3), axis=0)
        # print('state',self.observation )

        return np.array(self.observation).astype(np.float32)

    def get_pose_ob(self, ob_id):
        """
        get_pose_ob: returns only the pose of object ob_id
        """

        if (ob_id == 1):
            state_object_pos, state_object_or = p.getBasePositionAndOrientation(self.objectid1)  # 3D pos object;
        elif ob_id == 2:
            state_object_pos, state_object_or = p.getBasePositionAndOrientation(self.objectid2)  # 3D pos object;
        else:
            state_object_pos, state_object_or = p.getBasePositionAndOrientation(self.objectid3)  # 3D pos object;

        return state_object_pos, state_object_or

    def check_box(self):
        """
        check_box: checks if one of the objects has gone out of the reachable region
        """

        for i in range(1, 4):
            pos, orientation = self.get_pose_ob(i)

            if (pos[0] < e_x[0]) or (pos[0] > e_x[-1]) or (pos[1] < e_y[0]) or (pos[1] > e_y[-1]):
                print('Error position - out of region')
                return 1
            else:
                # print('c',np.max([abs(ele) for ele in p.getEulerFromQuaternion(orientation)]))
                if np.max([abs(ele) for ele in p.getEulerFromQuaternion(orientation)]) > 0.43:  # c.a. 25degree
                    print('Error orientation')
                    return 1

        return 0

    def close(self):
        """
        close: deletes the environment
        """

        # p.stopStateLogging(self.logId)
        p.disconnect()
