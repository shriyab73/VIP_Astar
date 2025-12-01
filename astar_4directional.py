import heapq
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

Coord = Tuple[int, int]


def heuristic(a: Coord, b: Coord) -> int:
    # 4 directions -> manhattan
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar_4dir(grid: List[List[int]], start: Coord, goal: Coord) -> Optional[List[Coord]]:
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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    open_heap: List[Tuple[float, Coord]] = []
    heapq.heappush(open_heap, (0, start))

    origin: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0}
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

        for dr, dc in directions:
            nr = current[0] + dr
            nc = current[1] + dc
            neighbor = (nr, nc)
            if not in_bounds(nr, nc):
                continue
            if grid[nr][nc] == 1:
                continue
            candidate_g = current_g + 1
            if candidate_g < g_score.get(neighbor, float("inf")):
                origin[neighbor] = current
                g_score[neighbor] = candidate_g
                f_score[neighbor] = candidate_g + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    open_set.add(neighbor)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    return None
