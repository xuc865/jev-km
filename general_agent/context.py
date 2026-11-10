"""Model views omit duplicated Runtime transport data, preserving public facts."""
from copy import deepcopy


def prompt_state(state: dict) -> dict:
    result = deepcopy(state)
    if isinstance(result.get('runtime'), dict):
        # The authoritative public state is in runtime. fast_state is its inner
        # projection; _dom repeats screen/candidates plus executor locators.
        result.pop('fast_state', None)
        result['runtime'].pop('_dom', None)
    return result
