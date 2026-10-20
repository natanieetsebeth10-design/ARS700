"""
Component A - Step 2: Motion Planning and Pathfinding

This script creates a grid of the arena and finds a path from the
start to the goal using Dijkstra's and A* algorithms.

It compares both algorithms based on the number of nodes explored,
runtime and path length. It also displays and saves the results.
"""

import heapq
import time
import numpy as np
import matplotlib.pyplot as plt


# Set the arena size and grid resolution.
ARENA_SIZE = 3.0
RESOLUTION = 0.05
GRID_N = int(ARENA_SIZE / RESOLUTION)

# Set the robot size and safety distance around obstacles.
ROBOT_RADIUS = 0.035
SAFETY_MARGIN = 0.025
INFLATION = ROBOT_RADIUS + SAFETY_MARGIN

# Store the positions and sizes of the obstacles.
OBSTACLES = [
    (0.5, 0.3, 0.10, 0.10),    # Red box.
    (-0.4, 0.6, 0.12, 0.12),   # Green cylinder.
    (0.2, -0.5, 0.075, 0.20),  # Blue box.
    (-0.7, -0.3, 0.15, 0.075), # Yellow box.
]

# Set the robot's starting point and goal.
START_WORLD = (-1.0, -1.0)
GOAL_WORLD = (1.0, 1.0)


def world_to_grid(x, y):
    # Convert world coordinates into grid coordinates.
    gx = int((x + ARENA_SIZE / 2) / RESOLUTION)
    gy = int((y + ARENA_SIZE / 2) / RESOLUTION)
    return gx, gy


def grid_to_world(gx, gy):
    # Convert grid coordinates back into world coordinates.
    x = gx * RESOLUTION - ARENA_SIZE / 2 + RESOLUTION / 2
    y = gy * RESOLUTION - ARENA_SIZE / 2 + RESOLUTION / 2
    return x, y


def build_occupancy_grid():
    """
    Create a grid showing where the robot can move.
    0 means the cell is free.
    1 means the cell contains an obstacle.
    """

    grid = np.zeros((GRID_N, GRID_N), dtype=np.uint8)

    # Check each cell to see if it is occupied by an obstacle.
    for gx in range(GRID_N):
        for gy in range(GRID_N):
            wx, wy = grid_to_world(gx, gy)

            for (ox, oy, hw, hh) in OBSTACLES:
                if (ox - hw - INFLATION <= wx <= ox + hw + INFLATION) and \
                   (oy - hh - INFLATION <= wy <= oy + hh + INFLATION):
                    grid[gx, gy] = 1
                    break

    return grid


def dijkstra(grid, start, goal):
    """
    Find the shortest path by checking the closest cells first.
    The robot can move up, down, left and right.
    """

    t0 = time.perf_counter()

    # Store the shortest distance found for each cell.
    dist = {start: 0.0}

    # Remember the previous cell to rebuild the path.
    prev = {}

    # Keep track of cells that have already been visited.
    visited = set()

    # Start with the starting cell.
    pq = [(0.0, start)]
    nodes_explored = 0

    # Allow movement in four directions.
    neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while pq:
        d, current = heapq.heappop(pq)

        # Skip cells that have already been visited.
        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        # Stop searching when the goal is reached.
        if current == goal:
            break

        cx, cy = current

        # Check the neighbouring cells.
        for dx, dy in neighbours:
            nx, ny = cx + dx, cy + dy

            # Only allow cells inside the grid that are not blocked.
            if 0 <= nx < GRID_N and 0 <= ny < GRID_N and grid[nx, ny] == 0:
                neighbour = (nx, ny)

                # Moving to a neighbouring cell costs 1.
                new_dist = d + 1.0

                # Update the distance if a shorter route is found.
                if new_dist < dist.get(neighbour, float("inf")):
                    dist[neighbour] = new_dist
                    prev[neighbour] = current
                    heapq.heappush(pq, (new_dist, neighbour))

    runtime = time.perf_counter() - t0

    # Return no path if the goal cannot be reached.
    if goal not in prev and goal != start:
        return None, nodes_explored, runtime

    # Follow the previous cells backwards to rebuild the path.
    path = [goal]

    while path[-1] != start:
        path.append(prev[path[-1]])

    # Reverse the path so it starts at the starting point.
    path.reverse()

    return path, nodes_explored, runtime


def euclidean(a, b):
    # Calculate the straight-line distance between two cells.
    return np.hypot(a[0] - b[0], a[1] - b[1])


def astar(grid, start, goal):
    """
    Find a path using the distance travelled and the estimated
    distance remaining to the goal.
    """

    t0 = time.perf_counter()

    # Store the distance travelled from the start.
    g_score = {start: 0.0}

    # Estimate the total cost using Euclidean distance.
    f_score = {start: euclidean(start, goal)}

    # Remember the previous cell to rebuild the path.
    prev = {}

    # Keep track of cells that have already been visited.
    visited = set()

    # Start with the starting cell and its estimated cost.
    pq = [(f_score[start], start)]
    nodes_explored = 0

    # Allow movement in four directions.
    neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while pq:
        _, current = heapq.heappop(pq)

        # Skip cells that have already been visited.
        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        # Stop searching when the goal is reached.
        if current == goal:
            break

        cx, cy = current

        # Check the neighbouring cells.
        for dx, dy in neighbours:
            nx, ny = cx + dx, cy + dy

            # Ignore cells outside the grid and cells containing obstacles.
            if 0 <= nx < GRID_N and 0 <= ny < GRID_N and grid[nx, ny] == 0:
                neighbour = (nx, ny)

                # Calculate the distance travelled using this cell.
                tentative_g = g_score[current] + 1.0

                # Keep the route if it is shorter than the previous one.
                if tentative_g < g_score.get(neighbour, float("inf")):
                    g_score[neighbour] = tentative_g

                    # Add the estimated remaining distance to the cost.
                    f_score[neighbour] = tentative_g + euclidean(neighbour, goal)

                    # Save the previous cell and add the new route to the queue.
                    prev[neighbour] = current
                    heapq.heappush(pq, (f_score[neighbour], neighbour))

    runtime = time.perf_counter() - t0

    # Return no path if the goal cannot be reached.
    if goal not in prev and goal != start:
        return None, nodes_explored, runtime

    # Follow the previous cells backwards to rebuild the path.
    path = [goal]

    while path[-1] != start:
        path.append(prev[path[-1]])

    # Reverse the path so it starts at the starting point.
    path.reverse()

    return path, nodes_explored, runtime


def path_length_metres(path):
    # Calculate the total path length in metres.
    length = 0.0

    for i in range(1, len(path)):
        x1, y1 = grid_to_world(*path[i - 1])
        x2, y2 = grid_to_world(*path[i])

        length += np.hypot(x2 - x1, y2 - y1)

    return length


if __name__ == "__main__":

    # Create the grid and convert the start and goal into grid coordinates.
    grid = build_occupancy_grid()
    start = world_to_grid(*START_WORLD)
    goal = world_to_grid(*GOAL_WORLD)

    # Run Dijkstra and record its results.
    dijkstra_path, dijkstra_nodes, dijkstra_time = dijkstra(grid, start, goal)

    # Run A* and record its results.
    astar_path, astar_nodes, astar_time = astar(grid, start, goal)

    # Calculate the length of each path if a path was found.
    dijkstra_len = path_length_metres(dijkstra_path) if dijkstra_path else None
    astar_len = path_length_metres(astar_path) if astar_path else None

    # Display the results so both algorithms can be compared.
    print("=" * 60)
    print("PATH PLANNING COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25}{'Dijkstra':<15}{'A*':<15}")
    print(f"{'Nodes explored':<25}{dijkstra_nodes:<15}{astar_nodes:<15}")
    print(f"{'Runtime (s)':<25}{dijkstra_time:<15.6f}{astar_time:<15.6f}")
    print(f"{'Path length (m)':<25}{dijkstra_len:<15.4f}{astar_len:<15.4f}")

    # Create the data folder and save the comparison results.
    import os

    os.makedirs("../data", exist_ok=True)

    with open("../data/planning_comparison.csv", "w") as f:
        f.write("metric,dijkstra,astar\n")
        f.write(f"nodes_explored,{dijkstra_nodes},{astar_nodes}\n")
        f.write(f"runtime_s,{dijkstra_time:.6f},{astar_time:.6f}\n")
        f.write(f"path_length_m,{dijkstra_len:.4f},{astar_len:.4f}\n")

    # Create two graphs to compare the paths.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    for ax, path, title, nodes, rt in [
        (axes[0], dijkstra_path, "Dijkstra's Algorithm", dijkstra_nodes, dijkstra_time),
        (axes[1], astar_path, "A* (Euclidean heuristic)", astar_nodes, astar_time),
    ]:

        # Display the grid and obstacles in the background.
        ax.imshow(
            grid.T, origin="lower", cmap="Greys",
            extent=[
                -ARENA_SIZE / 2, ARENA_SIZE / 2,
                -ARENA_SIZE / 2, ARENA_SIZE / 2
            ],
            alpha=0.6,
        )

        # Draw the path if the algorithm found one.
        if path:
            path_world = [grid_to_world(gx, gy) for gx, gy in path]
            xs, ys = zip(*path_world)
            ax.plot(xs, ys, linewidth=2.5, color="crimson")

        # Mark the starting point and the goal.
        ax.plot(*START_WORLD, marker="o", color="green", markersize=10, label="Start")
        ax.plot(*GOAL_WORLD, marker="*", color="blue", markersize=14, label="Goal")

        # Display the algorithm name, number of nodes and runtime.
        ax.set_title(f"{title}\nNodes explored: {nodes} | Runtime: {rt*1000:.2f} ms")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="lower right")
        ax.set_aspect("equal")

    # Adjust the graphs and save the comparison image.
    plt.tight_layout()
    plt.savefig("../data/path_comparison.png", dpi=150)
    plt.show()

    print("\nSaved: data/planning_comparison.csv and data/path_comparison.png")
