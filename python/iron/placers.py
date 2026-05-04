"""Back-compat shim: SequentialPlacer was removed in commit f2eae0d691;
tile placement now happens in MLIR --aie-place-tiles pass. This shim
exists ONLY so older IRON code that does `from aie.iron.placers import
SequentialPlacer` keeps working until IRON migrates. The class is a
no-op when passed to resolve_program (the new resolve_program signature
silently accepts it via *_args / **_kwargs)."""


class SequentialPlacer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None
