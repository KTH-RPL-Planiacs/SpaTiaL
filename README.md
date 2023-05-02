# SpaTiaL

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/KTH-RPL-Planiacs/SpaTiaL/python-app.yml?branch=main&style=for-the-badge)

SpaTiaL is a framework to specify spatial and temporal relations between objects.

## Installation

We use [MONA](http://www.brics.dk/mona/) to convert LTLf formulae to DFA. If you want to use the automaton-based planning, install it first.
Check the website for installation instructions or try to install with apt. **We are using `ltlf2dfa` to call MONA in python.
That library currently does not work with Windows.**
```shell
sudo apt install mona
```

## Repository Structure

- [spatial-lib](./spatial): source code for the library itself
- [experiments](./experiments): scripts to reproduce the experiments presented in our article
- [docs](./docs): Generated docs via [`Sphinx`](https://www.sphinx-doc.org/)
