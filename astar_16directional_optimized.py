import heapq
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

Coord = Tuple[int, int]


# ---------- Distance helpers ----------

def euclidean(a: Coord, b: Coord) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------- Dynamic weight coefficient ----------

def dynamic_weight(pos: Coord, anchor: Coord, initial_md: float) -> float:
    """
    Dynamic weight w(n) ~ 1 when close to anchor, >1 when far.
    We base it on the ratio of current Manhattan distance to initial distance.
    """
    d = manhattan(pos, anchor)
    if initial_md <= 0:
        return 1.0
    ratio = d / initial_md  # in [0, 1] if within bounding box

    # Smooth, bounded version inspired by the paper's exp + log idea
    w = 1.0 + math.exp(ratio) * math.log(ratio + 1.0)
    return w


# ---------- Line-of-sight collision check ----------

def line_is_clear(grid: List[List[int]], a: Coord, b: Coord) -> bool:
    """
    Check that the straight line from a to b does not cross obstacles.
    We sample along the segment and ensure all visited cells are free.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    (r0, c0), (r1, c1) = a, b

    if not in_bounds(r1, c1) or grid[r1][c1] == 1:
        return False

    dr = r1 - r0
    dc = c1 - c0
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


# ---------- 16-direction neighbors ----------

def neighbors_16dir(node: Coord, grid: List[List[int]]) -> List[Coord]:
    """
    16-direction “radius 2 ring” around the node:
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
    result: List[Coord] = []
    for dr, dc in directions_16:
        nr, nc = r + dr, c + dc
        if line_is_clear(grid, (r, c), (nr, nc)):
            result.append((nr, nc))
    return result


# ---------- Path helpers ----------

def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def prune_path(path: List[Coord], grid: List[List[int]]) -> List[Coord]:
    """
    Redundant waypoint deletion:
    Try to skip intermediate waypoints as long as line-of-sight is clear.
    """
    if not path or len(path) <= 2:
        return path

    pruned = [path[0]]
    anchor_idx = 0
    check_idx = 2

    while check_idx < len(path):
        anchor = path[anchor_idx]
        candidate = path[check_idx]
        if line_is_clear(grid, anchor, candidate):
            # we can skip the intermediate point(s); try an even farther point
            check_idx += 1
        else:
            # last one before candidate is the last safe waypoint
            last_safe_idx = check_idx - 1
            pruned.append(path[last_safe_idx])
            anchor_idx = last_safe_idx
            check_idx = anchor_idx + 2

    # always end at the goal
    if pruned[-1] != path[-1]:
        pruned.append(path[-1])

    return pruned


# ---------- Bidirectional 16-direction A* with dynamic weighting ----------

def bidirectional_astar_16dir(
    grid: List[List[int]],
    start: Coord,
    goal: Coord,
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

    initial_md = max(manhattan(start, goal), 1)

    # Forward search (start -> goal)
    open_f: List[Tuple[float, Coord]] = []
    heapq.heappush(open_f, (0.0, start))
    g_f: Dict[Coord, float] = {start: 0.0}
    came_from_f: Dict[Coord, Coord] = {}
    closed_f: set[Coord] = set()
    in_open_f: set[Coord] = {start}

    # Backward search (goal -> start)
    open_b: List[Tuple[float, Coord]] = []
    heapq.heappush(open_b, (0.0, goal))
    g_b: Dict[Coord, float] = {goal: 0.0}
    came_from_b: Dict[Coord, Coord] = {}
    closed_b: set[Coord] = set()
    in_open_b: set[Coord] = {goal}

    meeting: Optional[Coord] = None

    while open_f and open_b:
        # --- Expand one step forward ---
        if open_f:
            _, current_f = heapq.heappop(open_f)
            if current_f in closed_f:
                pass
            else:
                closed_f.add(current_f)
                in_open_f.discard(current_f)

                if current_f in closed_b or current_f in in_open_b:
                    meeting = current_f
                    break

                for nb in neighbors_16dir(current_f, grid):
                    cr, cc = current_f
                    nr, nc = nb
                    step_cost = euclidean((cr, cc), (nr, nc))
                    tentative_g = g_f[current_f] + step_cost
                    if tentative_g < g_f.get(nb, float("inf")):
                        g_f[nb] = tentative_g
                        came_from_f[nb] = current_f
                        w = dynamic_weight(nb, goal, initial_md)
                        f = tentative_g + w * euclidean(nb, goal)
                        heapq.heappush(open_f, (f, nb))
                        in_open_f.add(nb)

        # --- Expand one step backward ---
        if open_b:
            _, current_b = heapq.heappop(open_b)
            if current_b in closed_b:
                pass
            else:
                closed_b.add(current_b)
                in_open_b.discard(current_b)

                if current_b in closed_f or current_b in in_open_f:
                    meeting = current_b
                    break

                for nb in neighbors_16dir(current_b, grid):
                    cr, cc = current_b
                    nr, nc = nb
                    step_cost = euclidean((cr, cc), (nr, nc))
                    tentative_g = g_b[current_b] + step_cost
                    if tentative_g < g_b.get(nb, float("inf")):
                        g_b[nb] = tentative_g
                        came_from_b[nb] = current_b
                        w = dynamic_weight(nb, start, initial_md)
                        f = tentative_g + w * euclidean(nb, start)
                        heapq.heappush(open_b, (f, nb))
                        in_open_b.add(nb)

    if meeting is None:
        print("No path found.")
        return None

    # Reconstruct full path start -> meeting -> goal
    path_f = reconstruct_path(came_from_f, meeting)        # start -> meeting
    path_b = reconstruct_path(came_from_b, meeting)        # goal -> meeting
    # path_b is [goal ... meeting]; we want meeting->goal but without repeating meeting
    tail = list(reversed(path_b[:-1]))
    full_path = path_f + tail

    # Remove redundant waypoints
    pruned = prune_path(full_path, grid)
    return pruned


# ---------- Visualization ----------

def visualize_grid_and_path(
    grid: List[List[int]],
    path: Optional[List[Coord]] = None,
    start: Optional[Coord] = None,
    goal: Optional[Coord] = None,
    title: str = "Bidirectional 16-dir A* with Dynamic Weighting"
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


# ---------- Example usage ----------

if __name__ == "__main__":
    # 0 = free, 1 = obstacle
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (5, 7)

    path = bidirectional_astar_16dir(grid, start, goal)
    print("Pruned path:", path)
    visualize_grid_and_path(grid, path, start, goal)
