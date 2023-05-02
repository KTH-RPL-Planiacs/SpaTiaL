# SpaTiaL

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/KTH-RPL-Planiacs/SpaTiaL/python-app.yml?branch=main&style=for-the-badge)

SpaTiaL is a framework to specify spatial and temporal relations between objects.

## Installation

`spatial-spec` is distributed on PyPI.

We use [MONA](http://www.brics.dk/mona/) to convert LTLf formulae to DFA. If you want to use the automaton-based planning, install it first.
Check the website for installation instructions or try to install with apt. **We are using `ltlf2dfa` to call MONA in python.
That library currently does not work with Windows.**
```shell
sudo apt install mona
```

## Reproducing Paper Results

The experiments use [poetry](https://python-poetry.org/) to handle dependencies. Please make sure you have poetry installed.
Clone the repository and install dependencies:
```shell
git clone https://github.com/KTH-RPL-Planiacs/SpaTiaL.git
cd SpaTiaL
cd spatial-experiments
poetry install
```

You should now be able to run the planning examples:
```shell
poetry run planning_push
```
or 
```shell
poetry run planning_grasp
```


## Repository Structure

- [spatial-lib](./spatial-lib): source code for the library itself
- [spatial-experiments](./spatial-experiments): scripts to reproduce the experiments presented in our article
- [docs](./docs): Generated docs