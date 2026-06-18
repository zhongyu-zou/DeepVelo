from .train import *

from . import tool as tl
from . import plot as pl
from . import pipeline as pipe


def _patch_scvelo_parallelize():
    # scvelo's velocity_graph passes variable-length lists to parallelize, which
    # calls np.array(res) by default (as_array=True). This fails when neighbor
    # counts differ across cells. Force as_array=False in that module's namespace.
    #
    # scvelo.tools.__init__ does `from .velocity_graph import velocity_graph`,
    # which shadows the module attribute with the function of the same name.
    # We must fetch the actual module from sys.modules, not via attribute access.
    try:
        import sys
        import scvelo  # ensure scvelo is loaded so the module is in sys.modules
        _vg = sys.modules["scvelo.tools.velocity_graph"]
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


def _patch_scvelo_set_legend():
    # pandas 2.x made Series.cat.categories read-only; scvelo assigns to it directly.
    # Replace set_legend with a fixed copy that uses rename_categories instead.
    try:
        import sys
        import numpy as np
        import matplotlib.patheffects as patheffects
        import scvelo
        _pu = sys.modules.get("scvelo.plotting.utils")
        if _pu is None:
            import importlib
            _pu = importlib.import_module("scvelo.plotting.utils")

        def _fixed_set_legend(
            adata, ax, value_to_plot, legend_loc, scatter_array,
            legend_fontweight, legend_fontsize, legend_fontoutline,
            legend_align_text, groups,
        ):
            if legend_fontoutline is None:
                legend_fontoutline = 1
            obs_vals = adata.obs[value_to_plot]
            obs_vals = obs_vals.cat.rename_categories(obs_vals.cat.categories.astype(str))
            color_keys = adata.uns[f"{value_to_plot}_colors"]
            if isinstance(color_keys, dict):
                color_keys = np.array([color_keys[c] for c in obs_vals.cat.categories])
            valid_cats = np.where(obs_vals.value_counts()[obs_vals.cat.categories] > 0)[0]
            categories = np.array(obs_vals.cat.categories)[valid_cats]
            colors = np.array(color_keys)[valid_cats]

            if groups is not None:
                groups, _ = _pu.get_groups(adata, groups, value_to_plot)
                groups = [g for g in groups if g in categories]
                colors = [colors[list(categories).index(x)] for x in groups]
                categories = groups

            if legend_loc == "on data":
                legend_fontweight = "bold" if legend_fontweight is None else legend_fontweight
                texts = []
                for label in categories:
                    x_pos, y_pos = np.nanmedian(scatter_array[obs_vals == label, :], axis=0)
                    if isinstance(label, str):
                        label = label.replace("_", " ")
                    kw = dict(verticalalignment="center", horizontalalignment="center")
                    kw.update(dict(weight=legend_fontweight, fontsize=legend_fontsize))
                    pe = [patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")]
                    texts.append(ax.text(x_pos, y_pos, label, path_effects=pe, **kw))
                if legend_align_text:
                    autoalign = "y" if legend_align_text is True else legend_align_text
                    try:
                        from adjustText import adjust_text as adj_text
                        adj_text(texts, autoalign=autoalign, text_from_points=False, ax=ax)
                    except ImportError:
                        print("Please `pip install adjustText` for auto-aligning texts")
            else:
                for idx, label in enumerate(categories):
                    if isinstance(label, str):
                        label = label.replace("_", " ")
                    ax.scatter([], [], c=[colors[idx]], label=label)
                ncol = 1 if len(categories) <= 14 else 2 if len(categories) <= 30 else 3
                kw = dict(frameon=False, fontsize=legend_fontsize, ncol=ncol)
                if legend_loc == "upper right":
                    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), **kw)
                elif legend_loc == "lower right":
                    ax.legend(loc="lower left", bbox_to_anchor=(1, 0), **kw)
                elif "right" in legend_loc:
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), **kw)
                elif legend_loc != "none":
                    ax.legend(loc=legend_loc, **kw)

        _pu.set_legend = _fixed_set_legend
        # Any scvelo plotting module that used `from .utils import *` (or imported
        # set_legend by name) holds its own reference — patch every one found.
        for _mod in list(sys.modules.values()):
            mod_name = getattr(_mod, "__name__", "") or ""
            if mod_name.startswith("scvelo.plotting") and _mod is not _pu:
                if getattr(_mod, "set_legend", None) is not None:
                    _mod.set_legend = _fixed_set_legend
    except Exception:
        pass


_patch_scvelo_set_legend()
