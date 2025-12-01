import heapq
import math
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

Coord = Tuple[int, int]

def heuristic(a: Coord, b: Coord) -> float:
    # euclidean distance
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def line_is_clear(grid: List[List[int]], a: Coord, b: Coord) -> bool:
    rows = len(grid)
    cols = len(grid[0])

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    (r0, c0) = a
    (r1, c1) = b
    dr = r1 - r0
    dc = c1 - c0

    if not in_bounds(r1, c1) or grid[r1][c1] == 1:
        return False
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
    neighbors: List[Coord] = []
    for dr, dc in directions:
        nr = r + dr
        nc = c + dc
        if line_is_clear(grid, (r, c), (nr, nc)):
            neighbors.append((nr, nc))

    return neighbors

def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar_16dir(grid: List[List[int]], start: Coord, goal: Coord) -> Optional[List[Coord]]:
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
            continue
        open_set.remove(current)

        cr, cc = current
        current_g = g_score[current]

        for neighbor in neighbors(current, grid):
            nr, nc = neighbor
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

    return None
