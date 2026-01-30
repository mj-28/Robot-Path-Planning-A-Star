# Robot Path Planning A*

This project implements the **A\*** search algorithm for mobile robot path planning in a 2D grid-based environment.  
The task is to compute an optimal path from a start location to a goal location while avoiding obstacles, minimising traversal cost.

The project was completed as part of an **Artificial Intelligence Principles** module and focuses on the **search algorithm design** rather than robot hardware or control.

## Problem Description

- The environment is represented as a **2D grid**
- Each cell is either:
  - **Free space** (traversable)
  - **Obstacle** (blocked)
- The robot moves in the grid from:
  - **Start node (A)**
  - **Goal node (B)**
- The objective is to find the **lowest-cost path** using an informed search strategy

This setup mirrors real-world problems such as:
- Mobile robot navigation
- Autonomous vehicle route planning
- Game AI pathfinding

Eval function: 
f(n) = g(n) + h(n)

## Heuristic

The heuristic function estimates the distance from the current node to the goal using:
- **Manhattan distance** (for grid-based movement)

## Technologies Used

- Python
- NumPy
- Matplotlib for visualisation
