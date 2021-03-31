import argparse, os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import coloring
import plotting


def directory(_str):
    if not os.path.isdir(_str):
        raise ValueError(f"{_str} is not a directory.")
    return _str

descr = """Plot all cached errors in a directory."""
parser = argparse.ArgumentParser(description=descr)
parser.add_argument('-d', '--directory', dest='DIRECTORY', type=directory, default=".cache/gaussian_M6G1", help='Path to the directory for the cached errors.')
args = parser.parse_args()
DIRECTORY = args.DIRECTORY


def plot(_error, _axes, _color):
    cacheFile = f'{DIRECTORY}/{_error}.npz'
    try:
        z = np.load(cacheFile)
        if 'errors_exact' in z.keys():
            print(f"WARNING: Old format in file `{cacheFile}`. Please convert first.")
            return
        elif 'errors' in z.keys():
            sampleSizes, errors = z['sampleSizes'], z['errors']
        plotting.plot_quantiles(sampleSizes, errors, qrange=(0.15,0.85), num_quantiles=5, linewidth_fan=1, color=_color, axes=_axes, zorder=2)
        _axes.set_xlim(max(ax.get_xlim()[0], sampleSizes[0]), min(_axes.get_xlim()[1], sampleSizes[-1]))
        return True
    except:
        return False


BG = "xkcd:white"
C0 = "C0"
C1 = "C1"
C2 = "C2"
C3 = "C3"
# BG = coloring.bimosyellow
# C0 = coloring.mix(coloring.bimosred, 80)
# C1 = "xkcd:black"

fontsize = 10
geometry = {
    'top':    1,
    'bottom': 0,
    'left':   0,
    'right':  1,
    'wspace': 0.25,  # the default as defined in rcParams
    'hspace': 0.25  # the default as defined in rcParams
}
figshape = (1,1)
figsize = coloring.compute_figsize(geometry, figshape, 2)
fig,ax = plt.subplots(*figshape, figsize=figsize, dpi=300)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

mask = [False]*4
mask[0] = plot("sparse_error", ax, C0)
mask[1] = plot("bstt_error", ax, C1)
mask[2] = plot("tt_error_minimal_ranks", ax, C2)
mask[3] = plot("tt_error", ax, C3)

#NOTE: stacking multiple patches seems to be hard. This is the way seaborn displays such graphs
legend_elements = np.asarray([
    (Line2D([0], [0], color=coloring.mix(C0, 80), lw=1.5), "sparse"),
    (Line2D([0], [0], color=coloring.mix(C1, 80), lw=1.5), "block-sparse TT"),
    (Line2D([0], [0], color=coloring.mix(C2, 80), lw=1.5), "dense TT (rank 1)"),
    (Line2D([0], [0], color=coloring.mix(C3, 80), lw=1.5), "dense TT (rank 14)")
], dtype=object)[mask,:].T
ax.legend(*legend_elements, loc='upper right', fontsize=fontsize)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel(r"\# samples", fontsize=fontsize)
ax.set_ylabel(r"rel. error", fontsize=fontsize)

plt.subplots_adjust(**geometry)
os.makedirs("figures", exist_ok=True)
# problem = os.path.basename(os.path.normpath(DIRECTORY))
plt.savefig(f"figures/gaussian.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches="tight") # , transparent=True)
