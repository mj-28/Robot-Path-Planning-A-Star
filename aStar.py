import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from itertools import count
import psutil
import heapq

######### Environment ##########
# Parameters for the grid environment
GRID_SIZE = 10  # 10x10 grid
OBSTACLE_POSITIONS = [(2, 2), (2, 3), (2, 4), (5, 5), (6, 5), (7, 5), (8, 9), (3, 0)]
START_POS = (0, 0)  # Starting position of the robot
GOAL_POS = (9, 9)  # Goal position of the robot
goal_reached = False  # Global variable to track if the goal is reached

X = [0, 1, 0, -1]
Y = [-1, 0, 1, 0]

# Node class representing a grid cell
class Node:
    def __init__(self, obs, start, goal, x, y):
        self.obs = obs
        self.start = start
        self.goal = goal
        self.x = x
        self.y = y

# Environment class
class Environment:
    def __init__(self):
        self.startpos = START_POS
        self.goalpos = GOAL_POS
        self.grid = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                obs = (i, j) in OBSTACLE_POSITIONS
                start = (i, j) == START_POS
                goal = (i, j) == GOAL_POS
                self.grid.append(Node(obs, start, goal, i, j))
    
    def possible_moves(self, x, y):
        possible_moves = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in OBSTACLE_POSITIONS:
                possible_moves.append((nx, ny))
        return possible_moves


# Visualizer class
class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.grid(which='both')

    def draw_environment(self, env):
        # Clear the existing figure
        self.ax.clear()
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.grid(which='both')

        # Draw the grid
        for cell in env.grid:
            if cell.obs:  # Obstacle
                self.ax.add_patch(patches.Rectangle((cell.y, GRID_SIZE - 1 - cell.x), 1, 1, facecolor='black'))
            elif cell.start:  # Start
                self.ax.add_patch(patches.Rectangle((cell.y, GRID_SIZE - 1 - cell.x), 1, 1, facecolor='red', edgecolor='black'))
            elif cell.goal:  # Goal
                self.ax.add_patch(patches.Rectangle((cell.y, GRID_SIZE - 1 - cell.x), 1, 1, facecolor='green', edgecolor='black'))
            else:
                self.ax.add_patch(patches.Rectangle((cell.y, GRID_SIZE - 1 - cell.x), 1, 1, facecolor='yellow', edgecolor='gray'))

    def update_visualization(self, x, y):
        # Update the grid dynamically
        self.ax.add_patch(patches.Rectangle((y, GRID_SIZE - 1 - x), 1, 1, facecolor='purple', edgecolor='gray'))
        plt.pause(0.2)
        self.ax.add_patch(patches.Rectangle((y,GRID_SIZE - 1 - x), 1, 1, facecolor='blue', edgecolor='gray'))
        plt.draw()
        plt.pause(0.2)

    def reset(self):
        self.draw_environment(env)
        plt.draw()
        plt.pause(0.2)
        
######### End of Environment Cost #########
        

#Heuristic function (Manhattan distance due to only 4 movement possible)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + 0.001 * (abs(a[0] - START_POS[0]) + abs(a[1] - START_POS[1]))

# Tree Search Algorithm
def tree_search(env, vis):

    startTime = time.perf_counter() #perf counter used due to very quick processing time on algo sometimes
    maxTime = 120 #max amount of time the algo can run until it is stopped for loop detection 
    process = psutil.Process()
    memory = process.memory_info().rss #memory tool to figure how much memory is used

    counter = count() #counter for tie-breaker
    currentPos = START_POS
    pathLen = 0 #initiate path length counter
    vis.update_visualization(*currentPos)
    

    while currentPos != GOAL_POS:
        print(f"Visited Node: {currentPos}") #printing visited nodes to build path up again at the end as the nodes visited cannot be kept in memory
        
        #loop detections system if timer has ran out
        if time.perf_counter() - startTime > maxTime:
            runtime = time.perf_counter() - startTime
            memoryUsed = process.memory_info().rss - memory
            print(f"Tree Search stopped due to timeout and potential loop detection. Path Length: {pathLen}, Runtime: {runtime:.4f}s, Memory: {memoryUsed / 1024:.2f} KB")
            return
        
        #figuring out the neighbours and what moves are possible
        neighbours = env.possible_moves(*currentPos)

        #evaluating neighbours by heuristic only as visited nodes cannot be saved
        neighbours = sorted(neighbours, key=lambda n: (heuristic(n, GOAL_POS), next(counter)))
        currentPos = neighbours[0] #moving to best neighbour (essentially best move)
        pathLen += 1  #add to path length
        vis.update_visualization(*currentPos)

    #calculating performance metrics
    runtime = time.perf_counter() - startTime
    memoryUsed = process.memory_info().rss - memory
    print(f"Tree Search completed. Path Length: {pathLen}, Runtime: {runtime:.4f}s, Memory: {memoryUsed / 1024:.2f} KB")
    
# Graph Search Algorithm
def graph_search(env, vis):
    startTime = time.perf_counter() #start timer
    process = psutil.Process()
    memory = process.memory_info().rss #for memory usage 

    #priority queue
    #stores nodes as (f(n), g(n), (x, y))
    pQueue = []
    heapq.heappush(pQueue, (0 + heuristic(START_POS, GOAL_POS), 0, START_POS))

    #dict to store cost of each node
    gCost = {START_POS: 0}

    #used to reconstruct path visually
    #so the algo can run on its own without the vis impacting its outcome on performance
    #and the vis doesnt need to be commented out
    pathDict = {}

    while pQueue:
        #pop the node with the lowest f(n) from the priority queue
        _, g, currentPos = heapq.heappop(pQueue)

        #if the goal is reached
        if currentPos == GOAL_POS:
            runtime = time.perf_counter() - startTime
            memoryUsed = process.memory_info().rss - memory

            #reconstruct and visualise the path
            path = []
            while currentPos in pathDict:
                path.append(currentPos)
                currentPos = pathDict[currentPos]
            path.reverse()

            #print performance metrics
            print(f"A* Search completed. Path: {path}")
            print(f"A* Search completed. Path Length: {len(path)}, Runtime: {runtime:.4f}s, Memory: {memoryUsed / 1024:.2f} KB")
            for n in path:
                vis.update_visualization(*n)
            return

        #explores neighbouring positions and calculates the coste to each neighbour
        #it then adds the better cost neighbour (shortest path) to the priority queue based on f-score
        for neighbour in env.possible_moves(*currentPos):
            maybeG = g + 1  # Each step has a cost of 1

            #checks if this path to the neighbour is better than any previous one
            if neighbour not in gCost or maybeG < gCost[neighbour]:
                gCost[neighbour] = maybeG
                f = maybeG + heuristic(neighbour, GOAL_POS)
                heapq.heappush(pQueue, (f, maybeG, neighbour))
                pathDict[neighbour] = currentPos


# Main execution
env = Environment()
vis = Visualizer()

# Draw initial environment
vis.draw_environment(env)
plt.show(block=False)

#Tree Search
tree_search(env, vis)

#Graph Search
# graph_search(env, vis)

plt.show()
