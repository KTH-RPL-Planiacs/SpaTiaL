# SpaTiaL Specifications

SpaTiaL is a framework to specify spatial and temporal relations between objects.

We use [MONA](http://www.brics.dk/mona/) to convert LTLf formulae to DFA. If you want to use the automaton-based planning, install it first.
Check the website for installation instructions or try to install with apt. **We are using `ltlf2dfa` to call MONA in python.
That library currently does not work with Windows.**
```shell
sudo apt install mona
```

## Installation from Source

`spatial-spec` is distributed on PyPI, but you can also install from source.

1. Clone the repository and navigate into it:
    ```
    git clone https://github.com/KTH-RPL-Planiacs/SpaTiaL.git
    cd SpaTiaL
    cd spatial-lib
    ```
2. This project uses [poetry](https://python-poetry.org/) to handle dependencies. Please make sure you have poetry installed and run:
    ```
    poetry install
    ```
3. Try running the unit tests to see if everything works:
    ```
    poetry run python -m unittest discover
    ```
