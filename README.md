# SpaTiaL

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/KTH-RPL-Planiacs/SpaTiaL/python-app.yml?branch=main&style=for-the-badge)

SpaTiaL is a framework to specify spatial and temporal relations between objects.

## Installation

We use [MONA](http://www.brics.dk/mona/) to convert LTLf formulae to DFA. If you want to use the automaton-based planning, install it first.
Check the website for installation instructions or try to install with apt.
```shell
sudo apt install mona
```

1. Clone the repository and navigate into it:
    ```
    $ git clone https://github.com/KTH-RPL-Planiacs/SpaTiaL.git
    cd SpaTiaL
    ```
2. This project uses [poetry](https://python-poetry.org/) to handle dependencies. Please make sure you have poetry installed and run:
    ```
    poetry install
    ```
3. Try running the unit tests to see if everything works:
    ```
    poetry run python -m unittest discover
    ```

## Repository Structure

- [spatial](./spatial): source code for the library itself
- [tests](./tests): unittests for SpaTiaL
- [sim](./sim): pybullet simulator library for the pushing and the pick-and-place experiments
- [urdf](./urdf): 3D object data for the pybullet simulator
- [experiments](./experiments): scripts that run the experiments presented in the article
