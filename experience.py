"""Persist interaction traces and distill them into reusable facts.

Facts are condition → adjustment records, not prose. New lessons merge with
existing ones; conflicting then-clauses are resolved by support count.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from marks import distill, sanitize_symbol

OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
IF_KEYS = {"guide", "range", "blocked", "tracker", "prompt_visible", "prompt", "overlay", "repeat", "after"}
THEN_KEYS = {"prefer", "no_travel", "must_once", "ignore_prompt", "scan", "creep", "seconds_scale", "look_scale", "look", "travel"}
LOOK_VALUES = {"allow", "once", "stop"}
TRAVEL_VALUES = {"ahead", "probe", "stop"}
PRIORITY_VALUES = {"core", "high", "normal", "low"}


def priority_rank(raw) -> int:
    """Lower number = higher priority. core=0 … low=3."""
    if isinstance(raw, (int, float)):
        return max(0, min(3, int(raw)))
    text = str(raw or "normal").strip().lower()
    return {"core": 0, "high": 1, "normal": 2, "low": 3}.get(text, 2)


def normalize_priority(raw, *, default: str = "normal") -> str:
    if isinstance(raw, (int, float)):
        return {0: "core", 1: "high", 2: "normal", 3: "low"}.get(max(0, min(3, int(raw))), default)
    text = str(raw or default).strip().lower()
    return text if text in PRIORITY_VALUES else default


def sanitize_fact(fact: dict) -> dict | None:
    if not isinstance(fact, dict):
        return None
    if fact.get("kind") == "symbol" or (isinstance(fact.get("do") or fact.get("then"), dict) and (fact.get("do") or fact.get("then") or {}).get("role") in {"self", "goal"}):
        return sanitize_symbol(fact)
    source_when = fact.get("when") if isinstance(fact.get("when"), dict) else fact.get("if")
    source_do = fact.get("do") if isinstance(fact.get("do"), dict) else fact.get("then")
    predicate = {}
    for key, value in (source_when or {}).items():
        if key not in IF_KEYS:
            continue
        if key == "repeat":
            try:
                predicate[key] = max(1, int(value))
            except (TypeError, ValueError):
                continue
        elif key == "after":
            text = str(value).strip().lower()
            if text in {"look", "travel", "wait"}:
                predicate[key] = text
        else:
            predicate[key] = value
    then = {}
    for key, value in (source_do or {}).items():
        if key not in THEN_KEYS:
            continue
        if key in {"seconds_scale", "look_scale"}:
            try:
                # Allow >1 so drive/racer packs can lengthen holds via knowledge.
                then[key] = max(0.18, min(2.5, float(value)))
            except (TypeError, ValueError):
                continue
        elif key == "look":
            text = str(value).strip().lower()
            if text in LOOK_VALUES:
                then[key] = text
        elif key == "travel":
            text = str(value).strip().lower()
            if text in TRAVEL_VALUES:
                then[key] = text
                then["no_travel"] = text == "stop"
        elif key == "prefer":
            then[key] = bool(value)
        else:
            then[key] = bool(value)
    why = str(fact.get("why") or fact.get("because") or "").strip()
    if not predicate or not then:
        return None
    ident = str(fact.get("id") or fact.get("kind") or "supervise").strip()[:40] or "supervise"
    source = str(fact.get("source") or "learn")
    default_priority = "high" if source == "pack" else "normal"
    family = str(fact.get("family") or "").strip().lower()
    topic = str(fact.get("topic") or "").strip()
    out = {
        "id": ident,
        "if": predicate,
        "when": predicate,
        "then": then,
        "do": then,
        "why": why,
        "scope": "generic",
        "kind": str(fact.get("kind") or "supervise"),
        "priority": normalize_priority(fact.get("priority"), default=default_priority),
    }
    if family:
        out["family"] = family
    if topic:
        out["topic"] = topic
    if "always" in fact:
        out["always"] = bool(fact.get("always"))
    return out


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _key(predicate: dict) -> str:
    return json.dumps(predicate, sort_keys=True, ensure_ascii=False)


def _same_then(left: dict, right: dict) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def lessons(before: dict, act: dict, after: dict) -> list[dict]:
    """Structured outcomes of one interact-observe pair."""
    found: list[dict] = []
    guide0 = str(before.get("guide") or "none")
    guide1 = str(after.get("guide") or "none")
    if act.get("look") and guide0 in OPPOSITE and guide1 == OPPOSITE[guide0]:
        found.append({
            "id": "overshoot-look",
            "when": {"guide": guide0, "after": "look"},
            "do": {"look_scale": 0.65},
            "why": "one look flipped the guide to the opposite side, so the next look on this bearing must be shorter",
            "kind": "overshoot",
        })
    if act.get("move") and guide0 == "ahead" and guide1 == "behind":
        distance = str(before.get("range") or "far")
        found.append({
            "id": "overshoot-run",
            "when": {"guide": "ahead", "range": distance, "after": "travel"},
            "do": {
                "prefer": distance in {"near", "at"},
                "seconds_scale": 0.45 if distance in {"near", "at"} else 0.75,
            },
            "why": "travel from ahead landed behind the target, so later travel on this range must be shorter",
            "kind": "overshoot",
        })
    if before.get("range") in {"near", "at"} and act.get("move"):
        scale = 0.4 if before.get("range") == "near" else 0.25
        found.append({
            "id": "near-slow",
            "when": {"range": before.get("range"), "tracker": True},
            "do": {"prefer": False, "seconds_scale": scale},
            "why": "the tracker is already close, so travel uses the slower option and a shorter hold",
            "kind": "near",
        })
    if before.get("blocked") and after.get("blocked") and act.get("move"):
        found.append({
            "id": "wall-stop",
            "when": {"blocked": True, "after": "travel"},
            "do": {"travel": "stop", "scan": True},
            "why": "travel into a blocked frame left the path blocked, so stop travel and scan",
            "kind": "blocked",
        })
    if (before.get("prompt") or before.get("prompt_visible")) and act.get("once"):
        found.append({
            "id": "prompt-must",
            "when": {"prompt_visible": True},
            "do": {"must_once": True, "travel": "stop"},
            "why": "an on-screen prompt key is visible, so press it before any travel",
            "kind": "prompt",
        })
    found.extend(distill(before, act, after))
    return found


class Memory:
    def __init__(self, folder: Path, filename: str = "experience.json") -> None:
        self.folder = folder
        self.path = folder / filename
        self.events = folder / "events.jsonl"
        self.facts: list[dict] = []
        self.conflicts: list[dict] = []
        # Common facts stay dormant here until sync_common(family, loaded_topics).
        self.common_library: list[dict] = []
        self.pending = 0
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            self.facts = list(data.get("facts") or [])
            self.conflicts = list(data.get("conflicts") or [])

    def matching(self, see: dict) -> list[dict]:
        extra = {"repeat", "after", "view", "shape", "color", "family", "topic"}
        matched = []
        for fact in self.facts:
            if fact.get("kind") == "symbol":
                continue
            predicate = fact.get("when") or fact.get("if") or {}
            if all(see.get(key) == value for key, value in predicate.items() if key not in extra):
                matched.append(fact)
        # Core / high pack rules beat low-support learned noise.
        matched.sort(
            key=lambda fact: (
                priority_rank(fact.get("priority")),
                -int(fact.get("support") or 0),
                str(fact.get("id") or ""),
            )
        )
        return matched

    def symbols(self) -> list[dict]:
        return [fact for fact in self.facts if fact.get("kind") == "symbol"]

    def ingest(self, before: dict, act: dict, after: dict) -> list[dict]:
        incoming = [item for item in (sanitize_fact(fact) for fact in lessons(before, act, after)) if item]
        line = json.dumps({"at": _now(), "before": before, "act": act, "after": after, "lessons": incoming}, ensure_ascii=False)
        self.folder.mkdir(parents=True, exist_ok=True)
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if incoming:
            self.merge(incoming)
            self.pending += 1
            if self.pending >= 3:
                self.save()
                self.pending = 0
        return incoming

    def merge(self, incoming: list[dict]) -> None:
        index = {_key(fact.get("when") or fact.get("if") or {}): fact for fact in self.facts}
        for fact in incoming:
            predicate = fact.get("when") or fact.get("if") or {}
            key = _key(predicate)
            current = index.get(key)
            now = _now()
            if current is None:
                then = dict(fact.get("do") or fact.get("then") or {})
                source = str(fact.get("source") or "learn")
                record = {
                    "id": str(fact.get("id") or key),
                    "if": predicate,
                    "when": predicate,
                    "then": then,
                    "do": then,
                    "why": str(fact.get("why") or ""),
                    "scope": "generic",
                    "kind": str(fact.get("kind") or "learn"),
                    "source": source,
                    "priority": normalize_priority(
                        fact.get("priority"),
                        default="high" if source == "pack" else "normal",
                    ),
                    "support": 1,
                    "contradict": 0,
                    "updated": now,
                }
                if fact.get("looks"):
                    record["looks"] = dict(fact.get("looks") or {})
                self.facts.append(record)
                index[key] = record
                continue
            left_then = dict(current.get("do") or current.get("then") or {})
            right_then = dict(fact.get("do") or fact.get("then") or {})
            left_core = {key: value for key, value in left_then.items() if key not in {"seconds_scale", "look_scale"}}
            right_core = {key: value for key, value in right_then.items() if key not in {"seconds_scale", "look_scale"}}
            if _same_then(left_core, right_core):
                current["support"] = int(current.get("support") or 0) + 1
                current["updated"] = now
                current["when"] = current.get("if") or predicate
                current["do"] = current.get("then") or left_then
                current["scope"] = "generic"
                # Keep the stronger priority when two agreeing facts merge.
                if priority_rank(fact.get("priority")) < priority_rank(current.get("priority")):
                    current["priority"] = normalize_priority(fact.get("priority"))
                elif "priority" not in current:
                    current["priority"] = normalize_priority(
                        fact.get("priority"),
                        default="high" if current.get("source") == "pack" else "normal",
                    )
                if not current.get("why") and fact.get("why"):
                    current["why"] = str(fact.get("why"))
                for scale_name in ("seconds_scale", "look_scale"):
                    left_scale = left_then.get(scale_name)
                    right_scale = right_then.get(scale_name)
                    if isinstance(left_scale, (int, float)) and isinstance(right_scale, (int, float)):
                        current.setdefault("then", {})[scale_name] = (
                            left_scale * max(int(current["support"]) - 1, 1) + right_scale
                        ) / max(int(current["support"]), 1)
                        current["do"] = current.get("then") or left_then
                continue
            current["contradict"] = int(current.get("contradict") or 0) + 1
            support = int(current.get("support") or 0)
            contradict = int(current["contradict"])
            if contradict > support:
                self.conflicts.append({"kept": fact.get("id"), "dropped": current.get("id"), "if": predicate, "at": now})
                then = dict(fact.get("do") or fact.get("then") or {})
                # Pack/core rules are not displaced by a noisier learned contradiction.
                if (
                    priority_rank(current.get("priority")) < priority_rank(fact.get("priority"))
                    and str(current.get("source") or "") == "pack"
                ):
                    current["contradict"] = contradict
                    continue
                replacement = {
                    "id": str(fact.get("id") or current.get("id")),
                    "if": predicate,
                    "when": predicate,
                    "then": then,
                    "do": then,
                    "why": str(fact.get("why") or current.get("why") or ""),
                    "scope": "generic",
                    "kind": str(fact.get("kind") or current.get("kind") or "learn"),
                    "source": "learn",
                    "priority": normalize_priority(fact.get("priority"), default="normal"),
                    "support": contradict,
                    "contradict": support,
                    "updated": now,
                    "replaced": current.get("id"),
                }
                self.facts = [item for item in self.facts if item is not current]
                self.facts.append(replacement)
                index[key] = replacement

    def reload(self) -> None:
        if not self.path.is_file():
            return
        data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        # Pack facts only; common stays in common_library until sync_common.
        self.facts = [fact for fact in (data.get("facts") or []) if str(fact.get("source") or "") != "common"]
        self.conflicts = list(data.get("conflicts") or [])

    def overlay_common(self, path: Path, *, family: str = "", default_topic: str = "") -> None:
        """Register shared facts into common_library (dormant until sync_common)."""
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        owned = {
            str(fact.get("id") or "")
            for fact in self.facts
            if str(fact.get("source") or "") != "common"
        }
        # Replace prior library entries from this family/file family tag.
        fam = str(family or "").strip().lower()
        if fam:
            self.common_library = [
                item for item in self.common_library if str(item.get("family") or "") != fam
            ]
        for fact in data.get("facts") or []:
            item = sanitize_fact(fact)
            if not item:
                continue
            ident = str(item.get("id") or "")
            if ident and ident in owned:
                continue
            item["source"] = "common"
            item["family"] = str(item.get("family") or fam or "").strip().lower()
            item["topic"] = str(item.get("topic") or default_topic or "").strip()
            if "always" not in item:
                item["always"] = bool(fact.get("always"))
            item["support"] = max(1, int(fact.get("support") or 1))
            item["contradict"] = int(fact.get("contradict") or 0)
            item["updated"] = str(fact.get("updated") or _now())
            # Dedup by id inside library.
            self.common_library = [
                old for old in self.common_library if str(old.get("id") or "") != ident
            ]
            self.common_library.append(item)
        # Drop any previously activated common rows; caller must sync_common.
        self.facts = [fact for fact in self.facts if str(fact.get("source") or "") != "common"]

    def sync_common(self, family: str, loaded_topics: list[str] | None = None) -> list[str]:
        """Activate common facts for this family whose topic is seeded/loaded."""
        fam = str(family or "").strip().lower()
        loaded = {str(item) for item in (loaded_topics or []) if item}
        self.facts = [fact for fact in self.facts if str(fact.get("source") or "") != "common"]
        active_ids: list[str] = []
        for fact in self.common_library:
            fact_fam = str(fact.get("family") or "").strip().lower()
            if fact_fam and fam and fact_fam != fam:
                continue
            topic = str(fact.get("topic") or "").strip()
            always = bool(fact.get("always"))
            if always or (topic and topic in loaded) or (not topic and not fact_fam):
                self.facts.append(fact)
                active_ids.append(str(fact.get("id") or ""))
        return active_ids

    def ingest_facts(self, incoming: list[dict], source: str = "supervisor") -> list[dict]:
        cleaned = []
        for fact in incoming:
            item = sanitize_fact(fact)
            if not item:
                continue
            item["source"] = source
            cleaned.append(item)
        if cleaned:
            self.merge(cleaned)
            self.save()
        return cleaned

    def distill_successes(self, traces: list[list[dict]]) -> list[dict]:
        learned: list[dict] = []
        for events in traces:
            for event in events:
                before = event.get("before") if isinstance(event, dict) else None
                act = event.get("act") if isinstance(event, dict) else None
                after = event.get("after") if isinstance(event, dict) else None
                if before and act and after:
                    learned.extend(self.ingest(before, act, after))
        self.save()
        return learned

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        # Shared common/* facts are overlays; never write them into a title pack.
        pack_facts = [fact for fact in self.facts if str(fact.get("source") or "") != "common"]
        payload = {"facts": pack_facts, "conflicts": self.conflicts[-40:]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
