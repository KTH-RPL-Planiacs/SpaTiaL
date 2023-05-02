<h1 align="center">Pybullet Franka Emika Robot - Grasping task</h1>

This folder contains the code for the Grasping task simulation in PyBullet. The main files are:

1. [`sim_grasp.py`](./sim_grasp.py): provides an example of grasping simulation for the given formula

2. [`test.py`](./test.py): tests pick and place of the bottle in the area behind the Robot (positions randomly generated)

3. [`lib.py`](./lib.py): contains the definition of the ```PandaGraspEnv``` class, which has the following methods:

```python
    class PandaGraspEnv:
        def __init__(self,gui=0):
            """
            __init__: initializes the environment and set the parameters
            """
        def restore_joint(self):
            """
            restore_joint: restores initial joints position
            """
        def motor_control(self, pos, orn):
            """
            motor_control: computes inverse kinematics and performs motor control
            """
        def move(self,pos_goal,psi):
            """
            move: calls the functions to move the end-effector in pos_goal with desired
            orientation
            """
        def move_fingers(self, finger_target):
            """
            move_fingers: changes the fingers positions of the gripper
            """
        def rotate(self,rot):
            """
            rotate: rotate aroud the base joint
            """
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
        def grasp(self, ob_id, pos_ob2):
            """
            grasp: moves the object identified by ob_id in position pos_ob2
            """
        def reset(self,P,O):
            """
            reset: loads the robot, the objects and the environment and set the initial poses
            """
        def _update_constraints(self):
            """
            _update_constraints: updates internal constraints on joints and tools
            """
        def get_table_height(self):
            """
            Tableheight: returns the height of the table
            """
        def plot_traj(self,pos):
            """
            plot_traj: plots the trajectory tracked by end-effector
            """
        def _get_state(self):
            """
            _get_state: returns the current state
            """
        def restore_pose(self,pos,orient,ob_id):
            """
            restore_pose: restores the position and orientation of object ob_id
            """
        def check(self, POS, OR, OR_init):
            """
            check: checks if one of the objects has fallen or gone out of the reachable region
            """
        def check_success(self, pos, target):
            """
            check_success: checks if the object is in the target position
            """
        def print_statistics(self):
            """
            print_statistics: prints success/fail scores and number of executed actions
            """
        def close(self):
            """
            close: deletes the environment
            """
    ```

## Running code

    $ python3 sim_grasp.py

or

    $ python3 grasp_test2.py

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

* The Pybullet kitchen environment with articulated drawers is from [pybullet_kitchen](https://github.com/alphonsusadubredu/pybullet_kitchen)
  repository.
* Some of the items are from [OMG-Planner](https://github.com/alphonsusadubredu/pybullet_kitchen) repository.

## PyBullet

* [PyBullet Package](https://pypi.org/project/pybullet/)
* [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit)
