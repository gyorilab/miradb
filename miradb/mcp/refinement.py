"""Ontological refinement for model comparison.

Comparing concepts "by refinement" asks whether one grounding is an
ontological child of another (e.g. a specific infected state of a generic
one). MIRA answers this with a transitive closure of the DKG's
subclass/part-of relations, fetched from the DKG REST API set in
``MIRA_REST_URL``. The closure is large (~650k pairs, ~20 s to fetch), so
it is loaded once, in the background.
"""

import threading
from typing import Callable, Literal

from mira.metamodel.comparison import default_dkg_refinement_closure

RefinementMode = Literal["dkg", "exact"]


def _exact(child_curie: str, parent_curie: str) -> bool:
    """Refinement function under which no grounding refines another, so
    only equal concepts match."""
    return False


def prefetch_dkg_closure():
    """Start loading the DKG closure in a background thread."""
    threading.Thread(target=default_dkg_refinement_closure.initialize,
                     name="dkg-closure", daemon=True).start()


def get_refinement_func(
    mode: RefinementMode = "dkg",
) -> tuple[Callable[[str, str], bool], RefinementMode, str | None]:
    """Return a refinement function for ``mode``.

    Returns
    -------
    :
        The function, the mode actually used and a warning. When the DKG is
        unreachable, the mode falls back to ``exact`` and the warning says
        why, so results never silently claim ontology-aware matching.
    """
    if mode == "exact":
        return _exact, "exact", None
    try:
        default_dkg_refinement_closure.initialize()
    except Exception as err:
        return _exact, "exact", (
            f"DKG unavailable ({type(err).__name__}: {err}); compared by "
            "exact grounding only."
        )
    return default_dkg_refinement_closure.is_ontological_child, "dkg", None
