"""Input compilation to a model-selected target over currently visible empty tiles.

This does not rank rewards, choose encounters, open doors on the way, or inspect
other maps. Every consumable/encounter ends the path and requires a new choice.
"""
from collections import deque
from copy import deepcopy

DIRECTIONS = [('up', 0, -1), ('down', 0, 1), ('left', -1, 0), ('right', 1, 0)]


def reachable_targets(navigation: dict, keymap: dict) -> list[dict]:
    grid = navigation['grid']
    start = tuple(navigation['position'])
    walkable = set(navigation.get('walkable', ['.']))
    previous = {start: None}
    queue = deque([start])
    def tile(x, y):
        return grid[y][x] if 0 <= y < len(grid) and 0 <= x < len(grid[y]) else None
    while queue:
        x, y = queue.popleft()
        for direction, dx, dy in DIRECTIONS:
            point = (x + dx, y + dy)
            if point not in previous and tile(*point) in walkable:
                previous[point] = ((x, y), direction)
                queue.append(point)
    result = []
    for target in navigation.get('targets', []):
        if target.get('available') is False:
            continue
        goal = (target['x'], target['y'])
        choices = []
        for direction, dx, dy in DIRECTIONS:
            last = (goal[0] - dx, goal[1] - dy)
            if last not in previous:
                continue
            route = [(direction, goal)]
            point = last
            while point != start:
                parent, step_direction = previous[point]
                route.append((step_direction, point))
                point = parent
            route.reverse()
            choices.append(route)
        if not choices:
            continue
        route = min(choices, key=len)
        result.append({**target, 'keys': [keymap[direction] for direction, _ in route],
                       'positions': [list(point) for _, point in route], 'steps': len(route)})
    return result


def describe_reachability(navigation: dict, keymap: dict) -> dict:
    """Clarify visible targets vs executable paths without ranking a target."""
    result=deepcopy(navigation)
    executable={item['id'] for item in reachable_targets(navigation,keymap)}
    for target in result.get('targets',[]):
        target['resource_eligible']=target.get('available') is not False
        target['executable_now']=target['id'] in executable
    result['target_contract']=(
        'targets lists all currently visible map interactions. available/resource_eligible means only '
        'that resource or combat conditions allow contact, not that an empty-cell route reaches it. '
        'executable_now means eligible and reachable through empty visible cells. Other interactions '
        'stop navigation and require a separate model choice. Only the current candidate page is executable; '
        'revisiting a scene does not reset it or bypass blockers.')
    return result
