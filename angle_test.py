import math
import numpy as np
import time
import matplotlib.pyplot as plt


file_path = 'testout22.txt'

start_time = time.process_time()

# Read the file and preprocess it
with open(file_path, 'r') as file:
    lines = file.readlines()

# Remove brackets and split into numbers
cleaned_data = [line.replace('[', '').replace(']', '').strip() for line in lines]

# Convert to a NumPy array
occupancy_grid = np.array([list(map(int, line.split())) for line in cleaned_data])
print(np.sum(occupancy_grid))
print(occupancy_grid.shape)

def get_angle_difference(to_angle, from_angle):
    delta = to_angle - from_angle
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    return delta

def queue_based_exploration(occ_grid, cur_gps, goal_gps, orientation, 
                              latitude_length=111086.2, longitude_length=81978.2):
    cols, rows = occ_grid.shape[1], occ_grid.shape[0]
    robot_x, robot_y = 78, 47
    max_depth = 50
    waypoint_weight = 10.0
    waypoint_max_weight = 10.0
    
    delta_lat = goal_gps[0] - cur_gps[0]
    delta_lon = cur_gps[1] - goal_gps[1]
    north_m = delta_lat * latitude_length
    west_m = delta_lon * longitude_length
    
    # desired_heading_global = math.atan2(west_m, north_m)
    desired_heading_x = math.cos(orientation) * west_m + math.sin(orientation) * north_m
    desired_heading_y = -math.sin(orientation) * west_m + math.cos(orientation) * north_m
    desired_heading_global = math.atan2(desired_heading_y, desired_heading_x) - math.pi
    print("Hello")
    print(desired_heading_global)
    queue = set()
    queue.add((robot_x, robot_y))
    visted = set()
    best_pos = (robot_x, robot_y)
    best_cost = -float('inf')
    outside_point_x, outside_point_y = 78, 55
    depth = 0
    
    while depth < max_depth and queue:
        current_queue = list(queue)
        queue.clear()
        
        for x, y in current_queue:
            if (x, y) in visted:
                continue
            visted.add((x, y))
            
            if x < 0 or x >= cols or y < 0 or y >= rows:
                continue
            
            if occ_grid[y][x] != 1:
                continue

            euclidean_distance = math.sqrt((x - outside_point_x)**2 + (y - outside_point_y)**2)

            base_cost = euclidean_distance
            
            dx = x - outside_point_x
            dy = y - outside_point_y
            cell_dir_local = math.atan2(dy, dx)  
        
            global_cell_dir = orientation + cell_dir_local + math.pi * 0.5
            
            heading_error = abs(get_angle_difference(desired_heading_global, global_cell_dir))
            heading_error_deg = math.degrees(heading_error)
            # if x == 75 and y == 42:
            #     print(f"78,47's cell dir {math.degrees(global_cell_dir)}")
            # penalty = heading_error_deg

            penalty = min(heading_error_deg, waypoint_max_weight) * waypoint_weight
            total_cost = base_cost - penalty

            if total_cost > best_cost:
                best_cost = total_cost
                best_pos = (x, y)

            directions = [(-1, -1),  # Up-left (diagonal)
              (-1, 1),
              (1,1),
              (1,-1)]   # Right
            # directions = [(-1, 0),   # Up
            #   (-1, -1),  # Up-left (diagonal)
            #   (-1, 1),   # Up-right (diagonal)
            #   (0, -1),   # Left
            #   (0, 1),
            #   (1,0),
            #   (1,1),
            #   (1,-1)]  


            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visted:
                    queue.add((nx, ny))
        
        depth += 1
    print(f"best_cost: {best_cost}")
    print(f"best angle dif of {1 - best_cost}")
    return best_pos , desired_heading_global

current_gps = (42.668086, -83.218446)
goal_gps = (42.6679277, -83.2193276)
approx_north = (50.6009, -83.218446)


delta_lat_true_north = approx_north[0] - current_gps[0]
delta_lon_true_north = current_gps[1] - approx_north[1]
north_m_true_north = delta_lat_true_north * 111086.2
west_m_true_north = delta_lon_true_north * 81978.2

desired_heading_true_north = math.atan2(west_m_true_north, north_m_true_north)

robot_orientation_0 = math.radians(0)
robot_orientation_90 = math.radians(90)
robot_orientation_180 = math.radians(180)
robot_orientation_270 = math.radians(270)


best_cell, desired_heading_global = queue_based_exploration(
    occupancy_grid,
    current_gps,
    goal_gps,
    robot_orientation_270
)
print(f"Best cell to explore: {best_cell}")
print(f"Desired heading:  {math.degrees(desired_heading_global)}")
# Plot the occupancy grid
plt.figure(figsize=(10, 5))
plt.imshow(occupancy_grid, cmap='gray_r', origin='upper')
plt.scatter(78, 55, color='blue', label='Start Position')
plt.scatter(75, 42, color='pink', label='Testing point')

plt.scatter(best_cell[0], best_cell[1], color='red', label='Best Position')

# Plot vectors for desired_heading_global and robot_orientation
arrow_length = 10
# print(desired_heading_global)

# plt.arrow(78, 55, arrow_length * math.cos(desired_heading_true_north), arrow_length * math.sin(desired_heading_global), 
#           color='blue', head_width=2, label='Apporox North')
plt.arrow(78, 55, arrow_length * math.cos(robot_orientation_0), arrow_length * math.sin(robot_orientation_0), 
          color='red', head_width=2, label='North')
plt.arrow(78, 55, arrow_length * math.cos(robot_orientation_90), arrow_length * math.sin(robot_orientation_90), 
          color='orange', head_width=2, label='East')
plt.arrow(78, 55, arrow_length * math.cos(robot_orientation_180), arrow_length * math.sin(robot_orientation_180), 
          color='yellow', head_width=2, label='South')
plt.arrow(78, 55, arrow_length * math.cos(robot_orientation_270), arrow_length * math.sin(robot_orientation_270), 
          color='green', head_width=2, label='West')
plt.arrow(78, 55, arrow_length * math.cos(desired_heading_global), arrow_length * math.sin(desired_heading_global), 
          color='purple', head_width=2, label='Desired Heading')
plt.legend()
plt.title("Occupancy Grid with Start, Best Position, and Orientation Vectors")
plt.show()

