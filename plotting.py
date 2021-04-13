import numpy as np
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def mix(*color_value_list, alpha=True):
    if alpha:
        colorvec = lambda c: np.array(mpl.colors.to_rgba(c))
    else:
        colorvec = lambda c: np.array(mpl.colors.to_rgba(c))[:3]
        # colorvec = lambda c: np.array(mpl.colors.to_rgb(c))
    assert len(color_value_list) > 0
    c1 = colorvec(color_value_list[0])
    if len(color_value_list) == 1:
        return c1
    v = color_value_list[1]/100
    assert isinstance(v, float)
    if len(color_value_list) > 2:
        c2 = colorvec(color_value_list[2])
    else:
        c2 = 1
    c3 = v*c1 + (1-v)*c2
    return mix(c3, *color_value_list[3:])


bimosblack = "#23373B"
bimosred = "#A60000"  # (.65,0,0)
bimosyellow = "#F9F7F7"

# \setbeamercolor{normal text}{fg=mDarkTeal, bg=bimosyellow!50!bimosred!8}
# \setbeamercolor{alerted text}{fg=bimosred!80!bimosyellow}
# \setbeamercolor{example text}{fg=bimosred!50!bimosyellow!80}
normal_text_fg = bimosblack
normal_text_bg = mix(bimosyellow, 50, bimosred, 8)
alerted_text_fg = mix(bimosred, 80, bimosyellow)
example_text_fg = mix(bimosred, 50, bimosyellow, 80)
# \setbeamercolor{block title}{fg=normal text.fg, bg=normal text.bg!80!fg}
# \setbeamercolor{block body}{bg=block title.bg!50!normal text.bg}
block_title_bg = mix(normal_text_bg, 80, bimosblack)
block_body_bg = mix(block_title_bg, 50, normal_text_bg)


def set_BIMoS_style():
    mpl.rc('figure', facecolor=normal_text_bg, edgecolor=normal_text_bg)
    mpl.rc('axes', facecolor=block_body_bg, edgecolor=bimosblack, labelcolor=normal_text_fg)
    mpl.rc('xtick', color=normal_text_fg)
    mpl.rc('ytick', color=normal_text_fg)


def homotopy(x, y, num):
    assert x.shape in {(3,), (4,)}
    assert y.shape in {(3,), (4,)}
    x = x[None]
    y = y[None]
    s = np.linspace(0,1,num)[:,None]
    return (1-s)*x + s*y


def compute_figsize(geometry, shape, aspect_ratio=1, linewidth=3.98584):
    figwidth = linewidth
    subplotwidth = (geometry['right']-geometry['left'])*figwidth/(shape[1]+(shape[1]-1)*geometry['wspace'])  # make as wide as two plots
    subplotheight = subplotwidth/aspect_ratio
    figheight = subplotheight*(shape[0]+(shape[0]-1)*geometry['hspace'])
    figheight = figheight/(geometry['top']-geometry['bottom'])
    return (figwidth, figheight)


try:
    from skimage.color import rgb2lab as _rgb2lab, lab2rgb as _lab2rgb

    def rgb2lab(cs):
        assert cs.ndim >= 1
        assert cs.shape[-1] in {3,4}
        has_alpha = cs.shape[-1] == 4
        if has_alpha:
            alpha = cs[..., -1][..., None]
        ndim = cs.ndim
        cs = cs[(None,)*(3-ndim)][..., :3]
        ret = _rgb2lab(cs)[(0,)*(3-ndim)]
        if has_alpha:
            ret = np.concatenate([ret, alpha], axis=-1)
        return ret


    def lab2rgb(cs):
        assert cs.ndim >= 1
        assert cs.shape[-1] in {3,4}
        has_alpha = cs.shape[-1] == 4
        if has_alpha:
            alpha = cs[..., -1][..., None]
        ndim = cs.ndim
        cs = cs[(None,)*(3-ndim)][..., :3]
        ret = _lab2rgb(cs)[(0,)*(3-ndim)]
        if has_alpha:
            ret = np.concatenate([ret, alpha], axis=-1)
        return ret


    assert np.allclose(lab2rgb(rgb2lab(mix(bimosred))) - mix(bimosred), 0)


    def lightness(cs):
        return rgb2lab(cs)[..., 0]


    def sequential_cmap(*seq, lightness_range=(5, 98), steps=1000, strict=False):
        seq = np.array(sorted([rgb2lab(mix(c, alpha=False)) for c in seq], key=lambda c: c[0]))

        if seq[0][0] > lightness_range[0]:
            # if strict:
            #     raise ValueError("Color with minimum lightness not provided in mode 'strict'")
            c0 = seq[0].copy()
            c0[0] = lightness_range[0]
            seq = np.concatenate([c0[None], seq], axis=0)
        if seq[-1][0] < lightness_range[1]:
            # if strict:
            #     raise ValueError("Color with maximum lightness not provided in mode 'strict'")
            cm1 = seq[-1].copy()
            cm1[0] = lightness_range[1]
            seq = np.concatenate([seq, cm1[None]], axis=0)

        ratios = np.diff((seq[:,0] - lightness_range[0])/(lightness_range[1] - lightness_range[0]))
        parts = []
        for i in range(len(seq)-1):
            parts.append(homotopy(seq[i], seq[i+1], num=int(steps*ratios[i])))
        return lab2rgb(np.concatenate(parts))


    lightness_range = 5, 95
    bb = rgb2lab(mix(bimosblack, alpha=False))
    bb[0] = lightness_range[0]
    bb = lab2rgb(bb)
    br = rgb2lab(mix(bimosred, alpha=False))
    br[0] = (lightness_range[0] + lightness_range[1])/2
    br = lab2rgb(br)
    by = rgb2lab(mix(bimosyellow, alpha=False))
    by[0] = lightness_range[1]
    by = lab2rgb(by)

    # cmap = sequential_cmap(bb, mix(bimosblack, 20, bimosred), mix(bimosred, 20, br), mix(br, 80, example_text_fg), bimosyellow, lightness_range=lightness_range, strict=True)
    cmap = sequential_cmap(bb, mix(bimosblack, 30, bimosred, 80, br), mix(bimosred, 20, br, 80, example_text_fg), mix(br, 80, example_text_fg), bimosyellow, lightness_range=lightness_range, strict=True)
    BIMoSmap = mpl.colors.LinearSegmentedColormap.from_list('BIMoS', cmap)
except:
    print("WARNING: Could not load skimage")
    pass


def plot_quantiles(nodes, values, qrange=(0,1), ax=None, num_quantiles=4, linewidth_fan=0, **kwargs):
    """
    Plot the quantiles for a stochastic process.

    Parameters
    ----------
    nodes : ndarray (shape: (n,))
        Nodes at which the process is measured (index set, plotted on the x-axis).
    values : ndarray (shape: (m,n))
        Realizations of the the process (plotted on the y-axis).
        Each row of values contains a different path of the stochastic process.
    qrange : float (2,)
        Quantile range to use. (Outer bounds are currently excluded.)
    ax : matplotlib.axes.Axes, optional
        The axis object used for plotting. (default: matplotlib.pyplot.gca())
    num_quantiles : int (>= 0), optional
        Number of quantiles to plot. (default: 4)
    linewidth_fan : float, optional
        Linewidth of the bounding lines of each quantile. (default: 0)
    """
    errors = values
    values = nodes
    assert values.ndim == 1
    assert len(values) == errors.shape[1]
    assert num_quantiles >= 0
    if ax is None:
        ax = plt.gca()

    if num_quantiles == 0:
        ps = np.full((1,), 0.5)
        alphas = np.full((1,), kwargs.get('alpha', 1))
    else:
        ps = np.linspace(0,1,2*(num_quantiles+1)+1)[1:-1]
        alphas = np.empty(num_quantiles)
        alphas[0] = 2*ps[0]
        for i in range(1, num_quantiles):
            alphas[i] = 2*(ps[i] - ps[i-1])/(1 - 2*ps[i-1])
        alphas *= kwargs.get('alpha', 1)

    qs = np.quantile(errors, ps, axis=0)
    zorder = kwargs.pop('zorder', 1)
    base_line, = ax.plot(values, qs[num_quantiles], zorder=zorder+num_quantiles, **kwargs)
    ls = [base_line]
    for e in range(num_quantiles):
        l = ax.fill_between(values, qs[e], qs[-1-e], color=base_line.get_color(), alpha=alphas[e], zorder=zorder+e, linewidth=linewidth_fan)
        ls.append(l)
    return ls, alphas
