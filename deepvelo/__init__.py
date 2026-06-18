from .train import *

from . import tool as tl
from . import plot as pl
from . import pipeline as pipe


def _patch_scvelo_parallelize():
    # scvelo's velocity_graph passes variable-length lists to parallelize, which
    # calls np.array(res) by default (as_array=True). This fails when neighbor
    # counts differ across cells. Force as_array=False in that module's namespace.
    try:
        import scvelo.tools.velocity_graph as _vg
        _orig = _vg.parallelize

        def _fixed(*args, **kwargs):
            kwargs["as_array"] = False
            try:
                return _orig(*args, **kwargs)
            except TypeError:
                kwargs.pop("as_array")
                return _orig(*args, **kwargs)

        _vg.parallelize = _fixed
    except Exception:
        pass


_patch_scvelo_parallelize()
