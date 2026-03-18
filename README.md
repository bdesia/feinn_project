# feinn_project
CEIA FIUBA - Final Project - Finite Element-Integrated Neural Network framework

## About this project

This repository explores the application of deep learning techniques for the resolution of solid mechanics problems.

It is divided in two stages:

1. The FEINN framework integrates finite element methods with neural networks to create a hybrid approach for solving partial differential equations (PDEs) and other computationally intensive problems. This project aims to leverage the strengths of FEM for spatial discretization and neural networks for adaptive learning, enabling efficient and accurate solutions for engineering and physics-based applications.

1. Devolpment of a data-driven Fourier Neural Operator (FNO) to be used as a subrogate constitutive material model for multiscale simulations. Training data is generated from finite element simulations.

This project was developed as part of the CEIA (Curso de Especialización en Inteligencia Artificial) at FIUBA (Facultad de Ingeniería, Universidad de Buenos Aires).

## Project structure

The project was structured as follows. 

```
├── notebooks                           # Jupyter notebooks for interactive data analysis or modelling 
│   ├── FEINN_example N                 # Sub-folder of notebooks used for FEINN approach testing case N
│   └── RVE_data_generation             # Sub-folder of notebooks used for FNO subrogate model.
│       ├── checkpoints                 # Here you must find HPO data and trained model configuration.
│       ├── data                        # Finite-Element solutions for training model. One file per case.
│       ├── master_data                 # Whole dataset in a single file, divided in train/val/test, with normalizers stats.
│       ├── meshes                      # FE meshes for data generation.
│       └── strain_histories            # Synthetic generated strain paths for data generation.
└── src                                 # Main source of the project
```

## Prerequisites

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
    Before running, you must set device for PyTorch installation (choose between `cuda` or `cpu`). Default: `cuda` version.
1. Activate the poetry environment (just in case):

    ```bash
    poetry env activate
    ```
1. Now, you're ready to run the notebooks and make your own simulations.

## Code implementation

- [src/batch_elements.py](src/batch_elements.py): batched-version of finite elements library.

- [src/conditions.py](src/conditions.py): Boundary conditions and External loads classes. 

- [src/data_generator.py](src/data_generator.py):  Classes for strain paths generation for training subrogate constitutive models.

- [src/elements.py](src/elements.py): single-version of finite elements library.

- [src/evaluator.py](src/evaluator.py): Classes for testing, evaluate and make comparisons betweens models.

- [src/materials.py](src/materials.py): Classes for materials constitutive models.

- [src/rve_analyzer.py](src/rve_analyzer.py): Classes for handle FNO subrrogate constitutive model. From data pre-proccesing until evaluation. It contains:
    - `RVEDataset` class: dataloader for master file raw date.
    - `DualEncoderFNO` class: Our FNO generic model.
    - `RVEInferencer` class: for making inferences with the trained model.
    - `RVEVisualizer` class: for plotting and compare results.

- [src/solver.py](src/solver.py): Solver classes. It contains:
    - `NFEA` class: Non-linear finite element code
    - `FEINN` class: Finite Element-integrated Neural Network

- [src/trainer.py](src/trainer.py): It contains some useful classes for training FEINN models like self-adative loss weighting.

- [src/utils.py](src/utils.py): It contains some useful classes for finite element analysis like shape functions and gauss points.


## Component Interaction Flow

TODO

## Roadmap

### 1st Part: Finite Element integrated Neural Network for Solid Mechanics
- ✅ Torch implementation of 2D Non-linear Finite Element code
- ✅ Neural network integration with FEM
- ✅ Benchmark cases

**BONUS**
- ⬜ Subrrogate NN constitutive model implementation

### 2nd Part: FNO for Multi-scale in Solid Mechanics
- ✅ Strain history generation using Gaussin Process with RBF kernel.
- ✅ Synthethic data generation via FEM.
- ✅ FNO subrrogate model implementation.
- ⬜ Integration of FNO subrrogate model into macro solver.
- ⬜ Benchmark cases
