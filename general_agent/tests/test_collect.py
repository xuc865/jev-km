"""Tests for the batch collector's game selection.

The collector itself needs a browser and a model, so what is tested here is the
part that decides *which* games are eligible and how the entry path is rewritten.
That decision is where a silent bug costs a whole batch: selecting a game whose
entry is missing produced a FileNotFoundError after the browser had already
started, and the previous run wrote a row with no steps for it.

The catalogue uses Chinese column names, so the tests build a small CSV with the
real headers rather than a stylised stand-in.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.collect_sft import runnable_games  # noqa: E402

HEADERS = ["游戏标识", "分类目录", "包内入口", "运行载体/引擎"]


def write_index(root: Path, rows: list[dict]) -> Path:
    path = root / "INDEX.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def touch(root: Path, relative: str) -> None:
    target = root / relative.replace("/", "/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("<!doctype html>", encoding="utf-8")


class SelectionTests(unittest.TestCase):
    def test_only_games_with_an_existing_index_are_selected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            touch(root, "h5/a/one/index.html")
            rows = [
                {"游戏标识": "one", "分类目录": "h5/a", "包内入口": "h5/a/one/index.html",
                 "运行载体/引擎": "html5"},
                {"游戏标识": "two", "分类目录": "h5/a", "包内入口": "h5/a/two/index.html",
                 "运行载体/引擎": "html5"},
            ]
            games = runnable_games(write_index(root, rows), root)
        self.assertEqual([game["id"] for game in games], ["one"])

    def test_non_html_entries_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            touch(root, "doom/run.bat")
            rows = [
                {"游戏标识": "doom", "分类目录": "gzdoom", "包内入口": "doom/run.bat",
                 "运行载体/引擎": "gzdoom"},
            ]
            games = runnable_games(write_index(root, rows), root)
        self.assertEqual(games, [])

    def test_catalogue_metadata_is_carried_through(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            touch(root, "h5/b/game/index.html")
            rows = [
                {"游戏标识": "game", "分类目录": "h5/b", "包内入口": "h5/b/game/index.html",
                 "运行载体/引擎": "html5/phaser"},
            ]
            games = runnable_games(write_index(root, rows), root)
        self.assertEqual(games[0]["category"], "h5/b")
        self.assertEqual(games[0]["description"], "html5/phaser")
        self.assertEqual(games[0]["entry"], "h5/b/game/index.html")

    def test_a_bom_prefixed_catalogue_is_read(self) -> None:
        # The real INDEX.csv is UTF-8 with a BOM; reading it without utf-8-sig
        # leaves the first header prefixed by U+FEFF and silently selects nothing.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            touch(root, "h5/c/x/index.html")
            rows = [{"游戏标识": "x", "分类目录": "h5/c",
                     "包内入口": "h5/c/x/index.html", "运行载体/引擎": "html5"}]
            games = runnable_games(write_index(root, rows), root)
        self.assertEqual([game["id"] for game in games], ["x"])

    def test_duplicate_ids_are_left_to_the_caller(self) -> None:
        # Selection stays a pure filter: deduplicating here would hide a catalogue
        # error instead of letting the batch show it as two runs of one game.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            touch(root, "h5/d/dup/index.html")
            rows = [
                {"游戏标识": "dup", "分类目录": "h5/d", "包内入口": "h5/d/dup/index.html",
                 "运行载体/引擎": "html5"},
                {"游戏标识": "dup", "分类目录": "h5/d", "包内入口": "h5/d/dup/index.html",
                 "运行载体/引擎": "html5"},
            ]
            games = runnable_games(write_index(root, rows), root)
        self.assertEqual(len(games), 2)


if __name__ == "__main__":
    unittest.main()
