# PACS- Challenge1

This repository contains C++ code implementing the solution of the first challenge of the 2023/2024 PACS course.

## Files

### `main.cpp`

This file contains the main code for performing optimization using gradient descent methods. 
- It defines a test function (the one provided in the challenge).
- It defines the gradient function (centered finite elements).
- It allows the user to choose optimization method and learning rate decay strategy.
- It performs optimization and prints the result.

### `gradient_descent.cpp`

This file implements the minimize method Standard GD, Momentum GD, and Nesterov GD for optimization.

### `gradient_descent.hpp`

This header file defines classes and structs used in optimization, including optimization parameters, learning rate decay strategies, and optimization methods.

The structures selected largely employ polymorphism. 

## Usage

To run the optimization program:

1. Run `make` to compile the project: this command will exploit the Makefile for compilation and linking stages.
3. run the executable: `./main`

The program will prompt you to choose the learning rate decay strategy and optimization method. Press enter to use default options.

