"""Back-compat shim: placeable.py was removed in commit f2eae0d691.
No current IRON code imports from this module; kept as an empty
namespace so any straggler `from aie.iron.placeable import ...` import
yields a clean AttributeError rather than a ModuleNotFoundError."""
