import heapq
import math
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

Coord = Tuple[int, int]


# -------- Heuristic --------

def heuristic(a: Coord, b: Coord) -> float:
    """Euclidean distance heuristic."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# -------- Line-of-sight helper (avoid jumping 'through' obstacles) --------

def line_is_clear(grid: List[List[int]], a: Coord, b: Coord) -> bool:
    """
    Approximate collision check along the segment from a to b.
    Samples intermediate points and ensures all touched cells are free.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    (r0, c0), (r1, c1) = a, b
    dr = r1 - r0
    dc = c1 - c0

    # Always require end cell to be free and in bounds
    if not in_bounds(r1, c1) or grid[r1][c1] == 1:
        return False

    # Sample along the segment at sub-cell resolution
    steps = max(abs(dr), abs(dc)) * 2
    if steps == 0:
        return True

    for k in range(1, steps):
        t = k / steps
        rr = r0 + dr * t
        cc = c0 + dc * t
        r_i = int(round(rr))
        c_i = int(round(cc))
        if not in_bounds(r_i, c_i):
            return False
        if grid[r_i][c_i] == 1:
            return False

    return True


# -------- 16-direction neighbor function --------

def neighbors_16dir(node: Coord, grid: List[List[int]]) -> List[Coord]:
    """
    16-direction neighbors, inspired by the reference code.

    These are the “radius-2 ring” directions:
        (-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2),
        (-1, -2), (0, -2), (1, -2), (2, -2),
        (2, -1), (2, 0), (2, 1), (2, 2),
        (1, 2), (0, 2), (-1, 2)
    """
    directions_16 = [
        (-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2),
        (-1, -2), (0, -2), (1, -2), (2, -2),
        (2, -1), (2, 0), (2, 1), (2, 2),
        (1, 2), (0, 2), (-1, 2),
    ]

    r, c = node
    neighbors: List[Coord] = []

    for dr, dc in directions_16:
        nr, nc = r + dr, c + dc
        # use the line-of-sight check so this move doesn't cut through obstacles
        if line_is_clear(grid, (r, c), (nr, nc)):
            neighbors.append((nr, nc))

    return neighbors


# -------- Generic A* using the 16-direction neighbors --------

def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_16dir(
    grid: List[List[int]],
    start: Coord,
    goal: Coord
) -> Optional[List[Coord]]:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    if not in_bounds(*start) or not in_bounds(*goal):
        print("Start or goal out of bounds.")
        return None
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        print("Start or goal is blocked.")
        return None

    open_heap: List[Tuple[float, Coord]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    f_score: Dict[Coord, float] = {start: heuristic(start, goal)}

    open_set = {start}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            return reconstruct_path(came_from, current)

        if current not in open_set:
            continue  # stale
        open_set.remove(current)

        cr, cc = current
        current_g = g_score[current]

        for neighbor in neighbors_16dir(current, grid):
            nr, nc = neighbor
            # step cost is Euclidean distance for this jump
            dr = nr - cr
            dc = nc - cc
            step_cost = math.sqrt(dr * dr + dc * dc)
            tentative_g = current_g + step_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in open_set:
                    open_set.add(neighbor)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    print("No path found.")
    return None


# -------- Visualization (same style as before) --------

def visualize_grid_and_path(
    grid: List[List[int]],
    path: Optional[List[Coord]] = None,
    start: Optional[Coord] = None,
    goal: Optional[Coord] = None,
    title: str = "16-direction A* Path"
):
    grid_np = np.array(grid)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid_np, cmap="Greys", origin="upper")

    if path:
        ys = [r for (r, c) in path]
        xs = [c for (r, c) in path]
        plt.plot(xs, ys, linewidth=2, marker="o", markersize=4)

    if start:
        plt.scatter(start[1], start[0], marker="s", s=120,
                    edgecolors="green", facecolors="none", linewidths=2, label="Start")
    if goal:
        plt.scatter(goal[1], goal[0], marker="X", s=140,
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


# -------- Example usage --------

if __name__ == "__main__":
    # 0 = free, 1 = obstacle
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (5, 7)

    path = astar_16dir(grid, start, goal)
    print("Path:", path)
    visualize_grid_and_path(grid, path, start, goal)
