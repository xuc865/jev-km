"""Map catalogue/adapters smoke tests (no game launch)."""

from __future__ import annotations

import unittest

from general_agent.adapters.gzdoom import ACTIONS as GZ_ACTIONS
from general_agent.adapters.gzdoom import catalog_entry
from general_agent.adapters.html5 import ACTIONS as H5_ACTIONS
from general_agent.adapters.html5 import find_chrome


class MapCatalogTests(unittest.TestCase):
    def test_basic_is_gzdoom(self) -> None:
        entry = catalog_entry("basic")
        self.assertTrue(entry["entry_path"].endswith("run.bat"))
        self.assertTrue(any(action.id == "fire" for action in GZ_ACTIONS))

    def test_t_rex_is_html5(self) -> None:
        entry = catalog_entry("t-rex-runner")
        self.assertTrue(entry["entry_path"].endswith("index.html"))
        self.assertTrue(any(action.id == "jump" for action in H5_ACTIONS))

    def test_chrome_resolution_returns_something(self) -> None:
        self.assertTrue(find_chrome())


if __name__ == "__main__":
    unittest.main()
