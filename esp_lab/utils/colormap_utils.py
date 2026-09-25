import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap ## used to create custom colormaps
import matplotlib.colors as mcolors
import numpy as np

def blue2red_cmap(levels):
    """Discrete diverging color map with white in the middle, one color per interval.

    ``levels`` is the array of interval boundaries (preferred), or the number of
    boundaries of a range symmetric about zero. The intervals touching zero are
    white; negative intervals are blues and positive intervals yellow-to-red,
    each darkening away from zero. Use with ``BoundaryNorm(levels, ncolors=cmap.N)``.
    """
    if np.ndim(levels) == 0:
        nbins = int(levels) - 1
        mids = np.arange(nbins) - (nbins - 1) / 2.0
        half_width = 0.5
    else:
        levels = np.round(np.asarray(levels, dtype=float), 10)
        mids = 0.5 * (levels[:-1] + levels[1:])
        half_width = 0.5 * np.min(np.diff(levels))
    white = np.abs(mids) < 2.0 * half_width - 1e-9
    negative = (mids < 0) & ~white
    positive = (mids > 0) & ~white

    colors = np.ones((mids.size, 4))
    colors[negative] = plt.cm.Blues(np.linspace(1.0, 0.2, int(negative.sum())))
    colors[positive] = plt.cm.YlOrRd(np.linspace(0.15, 1.0, int(positive.sum())))
    return mcolors.ListedColormap(colors, name="blue2red")

def blue2red_acc_cmap(levs, cutoff):
    """Discrete ACC color map with exactly one color per interval of ``levs``.

    Intervals whose midpoint lies within one interval of zero are white
    (symmetric about zero); negative intervals are blues that darken away from
    zero; positive intervals are greys up to ``cutoff`` and yellow-to-red above
    it. Use with ``BoundaryNorm(levs, ncolors=cmap.N)`` so each color fills its
    whole colorbar interval.

    levs = ACC interval boundaries (e.g. -1.0 to 1.0 every 0.1)
    cutoff = positive value to highlight (e.g., 0.5)
    """
    levs = np.round(np.asarray(levs, dtype=float), 10)
    mids = 0.5 * (levs[:-1] + levs[1:])
    half_width = 0.5 * np.min(np.diff(levs))
    white = np.abs(mids) < 2.0 * half_width - 1e-9
    negative = (mids < 0) & ~white
    grey = (mids > 0) & ~white & (mids < cutoff)
    warm = (mids > 0) & ~white & (mids > cutoff)

    colors = np.ones((mids.size, 4))
    # Darkest at the most negative interval; the lightest blue stays visible.
    colors[negative] = plt.cm.Blues(np.linspace(1.0, 0.2, int(negative.sum())))
    colors[grey] = plt.cm.binary(np.linspace(0.15, 0.5, int(grey.sum())))
    # Start past YlOrRd's near-white end so the cutoff reads as a clear step.
    colors[warm] = plt.cm.YlOrRd(np.linspace(0.15, 1.0, int(warm.sum())))
    return mcolors.ListedColormap(colors, name="blue2red_acc")

def precip_cmap(n):
    """ combine two existing color maps to create a diverging color map with white in the middle.
    browns for negative, blues for positive
    n = the number of contour intervals
    """
    if (int(n/2) == n/2):
        # even number of contours
        nwhite=1
        nneg=n/2
        npos=n/2
    else:
        nwhite=2
        nneg = (n-1)/2
        npos = (n-1)/2

    colors1 = plt.cm.YlOrBr_r(np.linspace(0,1, int(nneg)))
    colors2 = plt.cm.GnBu(np.linspace(0,1, int(npos)))
    colorsw = np.ones((nwhite,4))

    colors = np.vstack((colors1, colorsw, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    return mymap

