import heapq
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

Coord = Tuple[int, int]

def euclidean(a: Coord, b: Coord) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def dynamic_weight(pos: Coord, anchor: Coord, initial_md: float) -> float:
    d = manhattan(pos, anchor)
    if initial_md <= 0:
        return 1.0
    ratio = d / initial_md
    w = 1.0 + math.exp(ratio) * math.log(ratio + 1.0)
    return w

def line_is_clear(grid: List[List[int]], a: Coord, b: Coord) -> bool:
    rows = len(grid)
    cols = len(grid[0])

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    (r0, c0) = a
    (r1, c1) = b

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

def neighbors(node: Coord, grid: List[List[int]]) -> List[Coord]:
    directions = [(-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (-1, 2)]
    r, c = node
    result: List[Coord] = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if line_is_clear(grid, (r, c), (nr, nc)):
            result.append((nr, nc))
    return result


def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def delete_redundant_waypoints(path: List[Coord], grid: List[List[int]]) -> List[Coord]:
    if not path or len(path) <= 2:
        return path

    no_redundant = [path[0]]
    anchor_index = 0
    check_index = 2

    while check_index < len(path):
        anchor = path[anchor_index]
        candidate = path[check_index]
        if line_is_clear(grid, anchor, candidate):
            check_index += 1
        else:
            last_safe_idx = check_index - 1
            no_redundant.append(path[last_safe_idx])
            anchor_index = last_safe_idx
            check_index = anchor_index + 2

    if no_redundant[-1] != path[-1]:
        no_redundant.append(path[-1])

    return no_redundant

def astar_16dir_optimized(grid: List[List[int]], start: Coord, goal: Coord) -> Optional[List[Coord]]:
    rows = len(grid)
    cols = len(grid[0])

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    if not in_bounds(*start) or not in_bounds(*goal):
        print("start or goal is out of bounds")
        return None
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        print("start or goal is blocked")
        return None

    initial_md = max(manhattan(start, goal), 1)

    # forward search
    open_f: List[Tuple[float, Coord]] = []
    heapq.heappush(open_f, (0.0, start))
    g_f: Dict[Coord, float] = {start: 0.0}
    came_from_f: Dict[Coord, Coord] = {}
    closed_f: set[Coord] = set()
    in_open_f: set[Coord] = {start}

    # backward search
    open_b: List[Tuple[float, Coord]] = []
    heapq.heappush(open_b, (0.0, goal))
    g_b: Dict[Coord, float] = {goal: 0.0}
    came_from_b: Dict[Coord, Coord] = {}
    closed_b: set[Coord] = set()
    in_open_b: set[Coord] = {goal}

    meeting: Optional[Coord] = None

    while open_f and open_b:
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

                for nb in neighbors(current_f, grid):
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

                for nb in neighbors(current_b, grid):
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
        return None

    path_f = reconstruct_path(came_from_f, meeting)
    path_b = reconstruct_path(came_from_b, meeting)
    tail = list(reversed(path_b[:-1]))
    full_path = path_f + tail
    no_redundant = delete_redundant_waypoints(full_path, grid)
    return no_redundant
