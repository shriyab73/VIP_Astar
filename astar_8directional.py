import heapq
import math
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

Coord = Tuple[int, int]

def heuristic(a: Coord, b: Coord) -> float:
    # costs 1 for straight moves, sqrt(2) for diagonal moves
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)
    #return dx + dy

def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar_8dir(grid: List[List[int]], start: Coord, goal: Coord) -> Optional[List[Coord]]:
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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    open_heap: List[Tuple[float, Coord]] = []
    heapq.heappush(open_heap, (0.0, start))

    origin: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    f_score: Dict[Coord, float] = {start: heuristic(start, goal)}

    open_set = {start}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            return reconstruct_path(origin, current)

        if current not in open_set:
            continue
        open_set.remove(current)

        current_g = g_score[current]

        r, c = current
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            neighbor = (nr, nc)

            if not in_bounds(nr, nc):
                continue
            if grid[nr][nc] == 1:
                continue

            step_cost = math.sqrt(2) if dr != 0 and dc != 0 else 1.0
            candidate_g = current_g + step_cost

            if candidate_g < g_score.get(neighbor, float("inf")):
                origin[neighbor] = current
                g_score[neighbor] = candidate_g
                f_score[neighbor] = candidate_g + heuristic(neighbor, goal)

                if neighbor not in open_set:
                    open_set.add(neighbor)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    return None

