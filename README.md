# SpaTiaL

SpaTiaL is a framework to specify spatial and temporal relations between objects.

## Installation

We use [MONA](http://www.brics.dk/mona/) to convert LTLf formulae to DFA. If you want to use the automaton-based planning, install it first.
Check the website for installation instructions or try to install with apt.
```shell
sudo apt install mona
```

### From Source

1. Clone the repository:
    ```
    $ git clone https://github.com/KTH-RPL-Planiacs/SpaTiaL.git
    ```
2. Before installing the required dependencies, you may want to create a virtual environment and activate it:
    ```
    $ virtualenv env
    $ source env/bin/activate
    ```
3. Install the dependencies necessary to run and test the code:
    ```
    $ pip install -r requirements.txt
    ```

## Repository Structure

- [spatial](./spatial): source code for the library itself
- [tests](./tests): unittests for SpaTiaL
- [sim](./sim): pybullet simulator library for the pushing and the pick-and-place experiments
- [urdf](./urdf): 3D object data for the pybullet simulator
- [experiments](./experiments): scripts that run the experiments presented in the article
