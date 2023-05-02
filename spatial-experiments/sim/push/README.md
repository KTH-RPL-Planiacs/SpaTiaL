<h1 align="center">Pybullet Franka Emika Robot - Pushing task</h1>

This folder contains the code for the Pushing task simulation in PyBullet. The main files are:

1. [`sim_push.py`](./sim_push.py): provides an example of pushing simulation

2. [`lib.py`](./lib.py): contains the definition of the ```PandaPushEnv``` class, which has the following methods:

```python
    class PandaPushEnv:
        def __init__(self,gui=0):
            """
            __init__: initializes the environment and set the parameters
            """
        def motor_control(self, pos, orn):
            """
            motor_control: computes inverse kinematics and performs motor control
            """
        def move_segment(self,pos_goal,psi,ob_id):
            """
            motor_control: moves end effector to pos_goal along the segment that link the current
            position to pos_goal, keeping psi-orientation
            """
        def move(self,pos_goal,psi):
            """
            move: calls the functions to move the end-effector in pos_goal with desired
            orientation
            """
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
        def track(self,path, ob_id):
            """
            track: moves the object ob_id along the path
            """
        def deletes_lines(self):
            """
            deletes_lines: removes all the debug lines stored in self.idsegment
            """
        def print_err(self):
            """
            print_err: prints statistics
            """
        def reset(self,objectid1Pos,objectid2Pos,objectid3Pos,objectid1Or,objectid2Or,objectid3Or):
            """
            reset: loads the robot, the objects and the environment and set the initial poses
            """
        def plot_path(self, path):
            """
            plot_path: plots debug lines to visualize the path
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
        def get_pose_ob(self,ob_id):
            """
            get_pose_ob: returns only the pose of object ob_id
            """
        def check_box(self):
            """
            check_box: checks if one of the objects has gone out of the reachable region
            """
        def close(self):
            """
            close: deletes the environment
            """
    ```

## Running code

    $ python3 sim_push.py

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

* The Pybullet kitchen environment with articulated drawers is from [pybullet_kitchen](https://github.com/alphonsusadubredu/pybullet_kitchen)
  repository.

## PyBullet

* [PyBullet Package](https://pypi.org/project/pybullet/)
* [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit)


