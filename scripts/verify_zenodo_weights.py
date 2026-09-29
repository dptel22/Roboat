"""Verify the Zenodo 12800597 checkpoint taxonomy straight from the .pt files.

The checkpoints were saved with a legacy ultralytics layout (`ultralytics.yolo.*`)
that current ultralytics cannot unpickle. We therefore unpickle with stub classes
substituted for every external symbol, then walk the reconstructed object tree
and print the real `names` / `nc` attributes of the saved model. Read-only.
"""
import io
import pickle
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "data" / "processed" / "zenodo_12800597" / "extracted" / "trained_weights"


class Stub:
    """Stand-in for any class the unpickler meets; records its state."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.state = None

    def __setstate__(self, state):
        self.state = state

def find(root, key):
    """Yield values for attribute/dict-entry `key` anywhere in the stub tree."""
    seen = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            if key in obj:
                yield obj[key]
            stack += list(obj.values())
        elif isinstance(obj, (list, tuple, set)):
            stack += list(obj)
        state = getattr(obj, "state", None)
        if isinstance(state, dict):
            if key in state:
                yield state[key]
            stack += list(state.values())
        d = getattr(obj, "__dict__", {})
        if isinstance(d, dict) and key in d:
            yield d[key]
        stack += [v for v in d.values()] if isinstance(d, dict) else []


_BUILTINS_ALLOW = {"dict", "list", "int", "str", "float", "tuple", "set", "bool", "bytes"}


def find_class(module, name):
    # Hardened: only plain builtin containers are real; everything else
    # (torch, ultralytics, builtins.callables) becomes an inert Stub, so the
    # pickle stream cannot reach eval/exec or construct arbitrary objects.
    if module == "builtins" and name in _BUILTINS_ALLOW:
        return getattr(__import__("builtins"), name)
    if module == "collections" and name in ("OrderedDict", "OrderdDict"):
        from collections import OrderedDict
        return OrderedDict
    return Stub


def persistent_load(saved_id):
    return None


for p in sorted(WEIGHTS.glob("*.pt")):
    print("=" * 60)
    print(p.name)
    with zipfile.ZipFile(p) as z:
        pkl_name = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        data = z.read(pkl_name)
    unpickler_cls = pickle._Unpickler  # pure-Python impl: find_class is overridable
    u = unpickler_cls(io.BytesIO(data))
    u.find_class = find_class
    u.persistent_load = persistent_load
    obj = u.load()
    found_names, found_nc = None, None
    for v in find(obj, "names"):
        if isinstance(v, dict) and v:
            found_names = v
    for v in find(obj, "nc"):
        if isinstance(v, int):
            found_nc = v
    print("  model.names =", found_names)
    print("  model.nc    =", found_nc)
    ok = found_names == {0: "ff_litter", 1: "hyacinth", 2: "ent_litter"} and found_nc == 3
    print("  VERDICT     :", "VERIFIED 3-class taxonomy (ff_litter/hyacinth/ent_litter)"
          if ok else "CHECK MANUALLY — unexpected taxonomy")
