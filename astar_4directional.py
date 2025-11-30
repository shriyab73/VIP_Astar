import heapq
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

Coord = Tuple[int, int]

# ---------- A* IMPLEMENTATION (4-DIR, MANHATTAN) ----------

def heuristic(a: Coord, b: Coord) -> int:
    """Manhattan distance heuristic for 4-directional grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar(
    grid: List[List[int]],
    start: Coord,
    goal: Coord
) -> Optional[List[Coord]]:
    """
    4-directional A* on a grid.

    grid: 2D list, 0 = free cell, 1 = obstacle
    start, goal: (row, col)
    Returns: list of coordinates from start to goal (inclusive) or None.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    if not in_bounds(*start) or not in_bounds(*goal):
        return None
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return None

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    open_heap: List[Tuple[float, Coord]] = []
    heapq.heappush(open_heap, (0, start))

    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0}
    f_score: Dict[Coord, float] = {start: heuristic(start, goal)}

    open_set = {start}

    while open_heap:
        current_f, current = heapq.heappop(open_heap)

        if current == goal:
            return reconstruct_path(came_from, current)

        if current not in open_set:
            # This is a stale entry; skip
            continue
        open_set.remove(current)

        current_g = g_score[current]

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            if not in_bounds(nr, nc):
                continue
            if grid[nr][nc] == 1:
                continue  # obstacle

            tentative_g = current_g + 1  # uniform cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in open_set:
                    open_set.add(neighbor)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    return None  # no path


# ---------- VISUALIZATION WITH MATPLOTLIB ----------

def visualize_grid_and_path(
    grid: List[List[int]],
    path: Optional[List[Coord]] = None,
    start: Optional[Coord] = None,
    goal: Optional[Coord] = None,
):
    """
    Visualize the obstacle grid and (optionally) the A* path.

    grid: 2D list, 0 = free, 1 = obstacle
    path: list of (row, col) coordinates returned by A*
    start, goal: optional markers
    """
    grid_np = np.array(grid)

    plt.figure(figsize=(6, 6))
    # Show obstacles (1) as black, free cells (0) as white
    plt.imshow(grid_np, cmap="Greys", origin="upper")

    # If we have a path, overlay it
    if path:
        # Path is in (row, col) => (y, x). For plotting, x is columns, y is rows.
        ys = [r for (r, c) in path]
        xs = [c for (r, c) in path]
        plt.plot(xs, ys, linewidth=2, marker="o", markersize=4)

    # Mark start and goal
    if start:
        plt.scatter(start[1], start[0], marker="s", s=100, edgecolors="green", facecolors="none", linewidths=2, label="Start")
    if goal:
        plt.scatter(goal[1], goal[0], marker="X", s=120, edgecolors="red", facecolors="none", linewidths=2, label="Goal")

    plt.title("Grid with Obstacles and A* Path")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.gca().invert_yaxis()  # row 0 at top like in matrix indexing

    # Show legend only if start/goal exist
    if start or goal:
        plt.legend(loc="upper right")

    plt.grid(False)
    plt.tight_layout()
    plt.show()


# ---------- EXAMPLE USAGE ----------

if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]

    start = (0, 0)
    goal = (4, 4)

    path = astar(grid, start, goal)
    print("Path:", path)

    visualize_grid_and_path(grid, path, start, goal)
