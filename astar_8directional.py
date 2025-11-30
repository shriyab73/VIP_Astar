import heapq
import math
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

Coord = Tuple[int, int]

# ---------- A* IMPLEMENTATION (8-DIR, OCTILE HEURISTIC) ----------

def heuristic(a: Coord, b: Coord) -> float:
    """
    Octile distance heuristic for 8-direction movement.
    Cost: 1 for straight moves, sqrt(2) for diagonals.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    #return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)
    return dx + dy

def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar_8dir(
    grid: List[List[int]],
    start: Coord,
    goal: Coord
) -> Optional[List[Coord]]:
    """
    8-directional A* on a grid.

    grid: 2D list, 0 = free cell, 1 = obstacle
    start, goal: (row, col)
    Returns: list of coordinates from start to goal (inclusive) or None.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    if not in_bounds(*start) or not in_bounds(*goal):
        print("Start or goal out of bounds")
        return None
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        print("Start or goal is blocked")
        return None

    # 8 directions: 4 cardinal + 4 diagonals
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),    # cardinal
        (-1, -1), (-1, 1), (1, -1), (1, 1)   # diagonals
    ]

    open_heap: List[Tuple[float, Coord]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    f_score: Dict[Coord, float] = {start: heuristic(start, goal)}

    open_set = {start}

    while open_heap:
        current_f, current = heapq.heappop(open_heap)

        if current == goal:
            return reconstruct_path(came_from, current)

        if current not in open_set:
            # stale heap entry
            continue
        open_set.remove(current)

        current_g = g_score[current]

        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            if not in_bounds(nr, nc):
                continue
            if grid[nr][nc] == 1:
                continue  # obstacle

            # Cost: 1 for straight, sqrt(2) for diagonal
            step_cost = math.sqrt(2) if dr != 0 and dc != 0 else 1.0
            tentative_g = current_g + step_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in open_set:
                    open_set.add(neighbor)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    print("No path found")
    return None


# ---------- VISUALIZATION WITH MATPLOTLIB ----------

def visualize_grid_and_path(
    grid: List[List[int]],
    path: Optional[List[Coord]] = None,
    start: Optional[Coord] = None,
    goal: Optional[Coord] = None,
    title: str = "8-dir A* Path"
):
    """
    Visualize the obstacle grid and (optionally) the A* path.

    grid: 2D list, 0 = free, 1 = obstacle
    path: list of (row, col) coordinates
    start, goal: optional markers
    """
    grid_np = np.array(grid)

    plt.figure(figsize=(6, 6))
    # Obstacles (1) = black, free (0) = white
    plt.imshow(grid_np, cmap="Greys", origin="upper")

    if path:
        ys = [r for (r, c) in path]
        xs = [c for (r, c) in path]
        plt.plot(xs, ys, linewidth=2, marker="o", markersize=4)

    if start:
        plt.scatter(start[1], start[0], marker="s", s=100,
                    edgecolors="green", facecolors="none", linewidths=2, label="Start")
    if goal:
        plt.scatter(goal[1], goal[0], marker="X", s=120,
                    edgecolors="red", facecolors="none", linewidths=2, label="Goal")

    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.gca().invert_yaxis()

    if start or goal:
        plt.legend(loc="upper right")

    plt.grid(False)
    plt.tight_layout()
    plt.show()


# ---------- EXAMPLE USAGE ----------

if __name__ == "__main__":
    # 0 = free, 1 = obstacle
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (5, 5)

    path = astar_8dir(grid, start, goal)
    print("Path:", path)
    print("Path length (steps):", len(path) if path else None)

    visualize_grid_and_path(grid, path, start, goal)
