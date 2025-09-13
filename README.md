# feinn_project
CEIA FIUBA - Final Project - Finite Element-Integrated Neural Network framework

## About this project

To complete

## Project structure

The project was structured as follows. 

```
├── data                                # Data. Here must be FE results used as ground-truth
├── notebooks                           # Jupyter notebooks for interactive data analysis or modeling 
│   └── Test                            # Sub-folder of notebooks used for code testing
└── src                                 # Main source of the project
    └── tests                           # Unit tests for all the packages in the source code
```

## Requirements

- Python > 3.11
- Poetry 2.1.4

## Running the project

### Locally (bash)

Follow this steps:
1. Clone the repository in your local machine.
1. Run `setup.sh` in a bash console (e.g. Git Bash). 

    ```bash
    ./setup.sh
    ```

    This script will execute poetry for installing all the dependencies and create a virtual environment. This script will aslso setup your `PYTHONPATH` by creating a `pth` file in the project virtual environment.
1. Activate the poetry environment (just in case):

    ```bash
    poetry env activate
    ```
