"""Learn what each minimap icon means by watching it after a look or a step.

The observer reports raw marks. Code assigns self/goal from what moved.
Colors and shapes are evidence, not a playbook.
"""

from __future__ import annotations

SHAPES = {"arrow", "diamond", "star", "dot", "other"}
COLORS = {"blue", "gold", "yellow", "white", "cyan", "other"}
BEARINGS = {"center", "left", "right", "ahead", "behind", "up", "down"}
ROLES = {"self", "goal"}
SIDES = {"left", "right", "ahead", "behind", "up", "down"}


def sanitize_marks(raw) -> list[dict]:
    found: list[dict] = []
    if not isinstance(raw, list):
        return found
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        shape = str(item.get("shape") or "other").strip().lower()
        color = str(item.get("color") or "other").strip().lower()
        bearing = str(item.get("bearing") or "center").strip().lower()
        if shape not in SHAPES:
            shape = "other"
        if color not in COLORS:
            color = "other"
        if bearing not in BEARINGS:
            bearing = "center"
        found.append({
            "shape": shape,
            "color": color,
            "bearing": bearing,
            "near": bool(item.get("near")),
        })
    return found


def sanitize_symbol(fact: dict) -> dict | None:
    if not isinstance(fact, dict):
        return None
    looks = fact.get("looks") if isinstance(fact.get("looks"), dict) else {}
    when = fact.get("when") if isinstance(fact.get("when"), dict) else fact.get("if") or {}
    then = fact.get("do") if isinstance(fact.get("do"), dict) else fact.get("then") or {}
    shape = str(looks.get("shape") or when.get("shape") or "").strip().lower()
    color = str(looks.get("color") or when.get("color") or "").strip().lower()
    role = str(then.get("role") or "").strip().lower()
    if shape not in SHAPES or color not in COLORS or role not in ROLES:
        return None
    predicate = {"view": "minimap", "shape": shape, "color": color}
    do = {"role": role}
    return {
        "id": str(fact.get("id") or f"minimap-{role}")[:40],
        "if": predicate,
        "when": predicate,
        "looks": {"shape": shape, "color": color},
        "then": do,
        "do": do,
        "why": str(fact.get("why") or "").strip(),
        "scope": "generic",
        "kind": "symbol",
    }


def symbols_of(memory) -> list[dict]:
    return [fact for fact in getattr(memory, "facts", []) if fact.get("kind") == "symbol"]


def _role(mark: dict, symbols: list[dict]) -> str:
    roles: set[str] = set()
    for fact in symbols:
        looks = fact.get("looks") or {}
        if looks.get("shape") == mark.get("shape") and looks.get("color") == mark.get("color"):
            role = str((fact.get("do") or fact.get("then") or {}).get("role") or "")
            if role in ROLES:
                roles.add(role)
    # Conflicting learned roles are evidence that this icon is not identified.
    # Treating an ambiguous icon as a goal is unsafe: it can send the agent over
    # a cliff or into a wall.
    if len(roles) != 1:
        return ""
    return next(iter(roles))


def _ambiguous(mark: dict, symbols: list[dict]) -> bool:
    roles = set()
    for fact in symbols:
        looks = fact.get("looks") or {}
        if looks.get("shape") == mark.get("shape") and looks.get("color") == mark.get("color"):
            role = str((fact.get("do") or fact.get("then") or {}).get("role") or "")
            if role in ROLES:
                roles.add(role)
    return len(roles) > 1


def resolve(marks: list[dict], symbols: list[dict]) -> tuple[dict | None, dict | None]:
    self_mark = next((item for item in marks if _role(item, symbols) == "self"), None)
    goal_mark = next((item for item in marks if _role(item, symbols) == "goal"), None)
    if self_mark is None:
        centered = [item for item in marks if item.get("bearing") == "center"]
        self_mark = next((item for item in centered if item.get("shape") == "arrow"), None)
    if goal_mark is None:
        others = [item for item in marks if item is not self_mark and item.get("bearing") in SIDES]
        goal_mark = next(
            (
                item for item in others
                if item.get("shape") in {"diamond", "star"}
                and _role(item, symbols) != "self"
                and not _ambiguous(item, symbols)
            ),
            None,
        )
    return self_mark, goal_mark


def guide_from(marks: list[dict], symbols: list[dict]) -> str:
    _self, goal = resolve(marks, symbols)
    if not goal:
        return "none"
    bearing = str(goal.get("bearing") or "none")
    if bearing == "center":
        return "ahead"
    return bearing if bearing in SIDES else "none"


def _pair(before: list[dict], after: list[dict], mark: dict) -> dict | None:
    return next(
        (
            item for item in after
            if item.get("shape") == mark.get("shape") and item.get("color") == mark.get("color")
        ),
        None,
    )


def _symbol(ident: str, mark: dict, role: str, why: str) -> dict:
    return {
        "id": ident,
        "kind": "symbol",
        "when": {"view": "minimap", "shape": mark["shape"], "color": mark["color"]},
        "looks": {"shape": mark["shape"], "color": mark["color"]},
        "do": {"role": role},
        "then": {"role": role},
        "why": why,
        "scope": "generic",
    }


def distill(before: dict, act: dict, after: dict) -> list[dict]:
    before_marks = before.get("marks") if isinstance(before.get("marks"), list) else []
    after_marks = after.get("marks") if isinstance(after.get("marks"), list) else []
    if not before_marks or not after_marks:
        return []
    found: list[dict] = []
    if act.get("look"):
        for mark in after_marks:
            if mark.get("bearing") == "center":
                found.append(_symbol(
                    "minimap-self",
                    mark,
                    "self",
                    "this mark stayed at the map center after a look, so it is the player",
                ))
        for mark in before_marks:
            if mark.get("bearing") not in SIDES:
                continue
            later = _pair(before_marks, after_marks, mark)
            if later and later.get("bearing") != mark.get("bearing"):
                found.append(_symbol(
                    "minimap-goal",
                    mark,
                    "goal",
                    "this mark moved around the player after a look, so it is the destination",
                ))
    if act.get("move"):
        for mark in before_marks:
            if mark.get("bearing") not in SIDES:
                continue
            later = _pair(before_marks, after_marks, mark)
            if not later:
                continue
            if later.get("bearing") == "ahead" or (later.get("near") and not mark.get("near")):
                found.append(_symbol(
                    "minimap-goal",
                    mark,
                    "goal",
                    "travel brought this mark ahead or closer, so it is the destination",
                ))
        for mark in after_marks:
            if mark.get("bearing") == "center":
                found.append(_symbol(
                    "minimap-self",
                    mark,
                    "self",
                    "this mark stayed at the map center after travel, so it is the player",
                ))
    return found
