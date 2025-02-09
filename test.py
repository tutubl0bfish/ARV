import numpy as np
import random
from collections import deque
import math
import matplotlib.pyplot as plt
import time
# Path to the file
file_path = 'testout2.txt'

start_time = time.process_time()


# Read the file and preprocess it
with open(file_path, 'r') as file:
    lines = file.readlines()

# Remove brackets and split into numbers
cleaned_data = [line.replace('[', '').replace(']', '').strip() for line in lines]

# Convert to a NumPy array
matrix = np.array([list(map(int, line.split())) for line in cleaned_data])
print(np.sum(matrix))
print(matrix.shape)


np.set_printoptions(threshold=np.inf,linewidth=1000)
start = (50, 78)

# Directions: Vertical, Horizontal, Diagonal
# directions = [(-1, 0),   # Up
#               (-1, -1),  # Up-left (diagonal)
#               (-1, 1),   # Up-right (diagonal)
#               (0, -1),   # Left
#               (0, 1),
#               (1,0),
#               (1,1),
#               (1,-1)]  

directions = [(-1, -1),  # Up-left (diagonal)
              (-1, 1),
              (1,1),
              (1,-1)]   # Right

# Function to check if a position is valid
def is_valid_move(position, matrix, visited):
    y, x = position
    rows, cols = matrix.shape
    return 0 <= y < rows and 0 <= x < cols and matrix[y, x] == 1 and position not in visited



def calculate_cost(start, current, rows, cols, edge_penalty_factor=0.2, distance_weight=0.09, min_distance=1e-6, start_penalty_factor=100):
    y_start, x_start = start
    y_current, x_current = current

    # Euclidean distance
    euclidean_distance = math.sqrt((x_current - x_start)**2 + (y_current - y_start)**2)
    
    # Ensure the distance is not zero when current == start
    if euclidean_distance == 0:
        euclidean_distance = min_distance  # Avoid zero distance

    # Apply weight to the distance
    weighted_distance = distance_weight * euclidean_distance

    # Find zero coordinates
    zero_coordinates = list(zip(*np.where(matrix == 0)))

    #Whats the closest obstacle
    distances_to_zeros = [math.sqrt((x_current - x_zero)**2 + (y_current - y_zero)**2) for y_zero, x_zero in zero_coordinates]
    min_distance_to_obstacle = min(distances_to_zeros)


    # Edge penalty
    edge_penalty = min(x_current, cols - x_current - 1, y_current, rows - y_current - 1)
    edge_penalty = max(0, edge_penalty)  # Ensure non-negative

    # Penalize if the current point is too close to the start
    if euclidean_distance <= min_distance:
        weighted_distance += start_penalty_factor  # Add penalty to move away from the start

    # Final cost
    cost = 4*(1/weighted_distance) + edge_penalty_factor * (1 / (edge_penalty + 1)) + 2*(1/(min_distance_to_obstacle+1))
    return cost

# This is used to get the cost from Maaz's new costmap. 
# This method relies on just getting the cost from the map.
# Depending on info on the map, may be necessary to alter existing cost function to not include distance to obstacle penalties.
def cost_from_new_map(start, current, rows, cols, edge_penalty_factor=0.2, distance_weight=0.09, min_distance=1e-6, start_penalty_factor=100, matrix_penalty_factor = 2):
    y_start, x_start = start
    y_current, x_current = current

    # Euclidean distance
    euclidean_distance = math.sqrt((x_current - x_start)**2 + (y_current - y_start)**2)
    
    # Ensure the distance is not zero when current == start
    if euclidean_distance == 0:
        euclidean_distance = min_distance  # Avoid zero distance

    # Apply weight to the distance
    weighted_distance = distance_weight * euclidean_distance

    # Cost from matrix/obstacles. May be more effcient to eliminate 0-values in the costmap first.
    #high: more obstacle-y
    obstacle_cost = max(matrix[y_current, x_current], min_distance)

    # Edge penalty
    edge_penalty = min(x_current, cols - x_current - 1, y_current, rows - y_current - 1)
    edge_penalty = max(0, edge_penalty)  # Ensure non-negative

    # Penalize if the current point is too close to the start
    if euclidean_distance <= min_distance:
        weighted_distance += start_penalty_factor  # Add penalty to move away from the start

    # Final cost
    cost = 4*(1/weighted_distance) + edge_penalty_factor * (1 / (edge_penalty + 1)) + matrix_penalty_factor * obstacle_cost
    return cost


# BFS Function
def bfs_with_cost(matrix, start):
    rows, cols = matrix.shape
    visited = set()
    queue = deque([start])
    visited.add(start)

    total_cost = 0
    cell_costs = []  # Store costs for analysis

    while queue:
        y, x = queue.popleft()

        # Calculate cost for this cell
        #cost = calculate_cost(start, (y, x), rows, cols)
        cost = cost_from_new_map(start, (y, x), rows, cols)
        total_cost += cost
        cell_costs.append(((y, x), cost))

        # Explore neighbors
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            # Update with new costmap: before, used matrix[ny, nx] == 1. 100 is the cost assigned to walls/obstacles
            if 0 <= ny < rows and 0 <= nx < cols and matrix[ny, nx] < 100 and (ny, nx) not in visited:
                queue.append((ny, nx))
                visited.add((ny, nx))

    return cell_costs, total_cost

def find_min_cost(cell_costs):
    min_cost = float('inf')  # Initialize to infinity
    min_cost_cell = None

    for cell, cost in cell_costs:
        if cost < min_cost:
            min_cost = cost
            min_cost_cell = cell

    return min_cost_cell, min_cost

def generate_cost_map(matrix, cell_costs):
    cost_map = np.full(matrix.shape, np.inf)  # Initialize cost map with infinity
    for (y, x), cost in cell_costs:
        cost_map[y, x] = cost  # Fill in the costs for visited cells
    return cost_map

# Visualize the cost map
def visualize_cost_map(cost_map):
    plt.figure(figsize=(10, 8))
    plt.imshow(cost_map, cmap='viridis', origin='upper')
    plt.colorbar(label='Cost')
    plt.title("Cost Heatmap")
    plt.xlabel("X-axis (Columns)")
    plt.ylabel("Y-axis (Rows)")
    plt.show()

def visualize_matrix_with_goal(matrix, start, goal):
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap="gray_r")  # Display matrix (1=white, 0=black)
    
    # Mark the start point
    plt.scatter(start[1], start[0], color="blue", label="Start", s=100)
    
    # Mark the optimal goal
    plt.scatter(goal[1], goal[0], color="red", label="Goal", s=100)
    
    # Add grid for clarity
    plt.grid(color="black", linestyle="--", linewidth=0.5)
    plt.xticks(range(matrix.shape[1]))
    plt.yticks(range(matrix.shape[0]))
    #plt.gca().invert_yaxis()  # Invert y-axis to align with matrix indexing
    plt.legend()
    plt.title("Matrix Visualization with Start and Goal")
    plt.show()


# Run BFS


cell_costs, total_cost = bfs_with_cost(matrix, start)

min_cost_cell, min_cost = find_min_cost(cell_costs)

print(f"\nCell with Minimum Cost: {min_cost_cell}, Minimum Cost: {min_cost:.2f}")

cost_map = generate_cost_map(matrix, cell_costs)
visualize_cost_map(cost_map)
visualize_matrix_with_goal(matrix, start, min_cost_cell)



# Print final matrix and path
# print("Final Matrix:")
# print(matrix)
# print("Path Traversed:")
# print(path)

print(matrix[27][102])

runtime = time.process_time() - start_time
print("Time: ", runtime)
