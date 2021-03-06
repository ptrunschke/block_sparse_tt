import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import plotting


PROBLEM = "darcy"
CACHE_DIRECTORY = ".cache/{problem}_M{order}G{maxGroupSize}"

order = 10
maxGroupSize = 3


CACHE_DIRECTORY = CACHE_DIRECTORY.format(problem=PROBLEM, order=order, maxGroupSize=maxGroupSize)
def plot(_error, _axes, _color):
    cacheFile = f'{CACHE_DIRECTORY}/{_error}.npz'
    try:
        z = np.load(cacheFile)
        sampleSizes, errors = z['sampleSizes'], z['errors']
        plotting.plot_quantiles(sampleSizes, errors, qrange=(0.15,0.85), num_quantiles=5, linewidth_fan=1, color=_color, axes=_axes, zorder=2)
        _axes.set_xlim(max(ax.get_xlim()[0], sampleSizes[0]), min(_axes.get_xlim()[1], sampleSizes[-1]))
    except:
        pass


BG = "xkcd:white"
C0 = "C0"
C1 = "C1"
C2 = "C2"
C3 = "C3"
# BG = plotting.bimosyellow
# C0 = plotting.mix(plotting.bimosred, 80)
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
figsize = plotting.compute_figsize(geometry, figshape, 2)
fig,ax = plt.subplots(*figshape, figsize=figsize, dpi=300)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

plot("sparse_error", ax, C0)
plot("bstt_error", ax, C1)
plot("bstt_sum_error", ax, C2)
plot("tt_error", ax, C3)

#NOTE: stacking multiple patches seems to be hard. This is the way seaborn displays such graphs
legend_elements = [
    (Line2D([0], [0], color=plotting.mix(C0, 80), lw=1.5), "$S_5^{10}$"),
    (Line2D([0], [0], color=plotting.mix(C1, 80), lw=1.5), "$S_{5,3}^{10}$"),
    (Line2D([0], [0], color=plotting.mix(C2, 80), lw=1.5), "$S_{5,3}^{10,\mathrm{aug}}$"),
    (Line2D([0], [0], color=plotting.mix(C3, 80), lw=1.5), "$T_{14}(V_6^{10})$")
]
legend = ax.legend(*zip(*legend_elements), loc='upper right', fontsize=fontsize)
legend.set_zorder(10)

ax.set_yscale('log')
ax.set_xscale('log')
# ax.set_xlabel(r"\# samples", fontsize=fontsize)
ax.set_xlabel(r"$M$", fontsize=fontsize)
ax.set_ylabel(r"relative error", fontsize=fontsize)

plt.subplots_adjust(**geometry)
os.makedirs("figures", exist_ok=True)
plt.savefig(f"figures/{PROBLEM}.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches="tight") # , transparent=True)
