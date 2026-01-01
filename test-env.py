# safe_env_introspect.py
from tmrl import get_environment
import numpy as np

MAX_DEPTH = 3

def print_attrs(obj, prefix="", depth=0, visited=None):
    """Safely print object attributes up to MAX_DEPTH and avoid cycles."""
    if visited is None:
        visited = set()

    if id(obj) in visited or depth > MAX_DEPTH:
        return
    visited.add(id(obj))

    for attr in dir(obj):
        if attr.startswith("__") and attr.endswith("__"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception as e:
            value = f"<ERROR: {e}>"

        print(f"{prefix}{attr}: {type(value)}")

        # Only recurse into objects that are not basic types
        if not isinstance(value, (int, float, str, bool, type(None), list, dict, tuple, set, np.ndarray)):
            print_attrs(value, prefix=prefix + "  ", depth=depth + 1, visited=visited)

if __name__ == "__main__":
    env = get_environment()
    print("Top-level env:")
    print_attrs(env)

    print("\nUnwrapped env:")
    try:
        unwrapped = env.unwrapped
    except AttributeError:
        unwrapped = env
    print_attrs(unwrapped)
