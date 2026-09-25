

def test_blue2red_acc_cmap_has_one_color_per_interval_and_symmetric_white():
    import numpy as np
    from matplotlib.colors import BoundaryNorm, to_hex

    from esp_lab.utils.colormap_utils import blue2red_acc_cmap

    levels = np.arange(-1.0, 1.0 + 0.05, 0.1)
    cmap = blue2red_acc_cmap(levels, 0.5)
    assert cmap.N == len(levels) - 1
    norm = BoundaryNorm(levels, ncolors=cmap.N)
    color = {round(m, 2): to_hex(cmap(norm(m))) for m in (levels[:-1] + levels[1:]) / 2}
    # White exactly for -0.1..0.1, and nowhere else.
    assert [m for m, c in color.items() if c == "#ffffff"] == [-0.05, 0.05]
    # Grey below the cutoff, warm above it: the step falls exactly at 0.5.
    r, g, b = (int(color[0.45][i:i + 2], 16) for i in (1, 3, 5))
    assert r == g == b
    r, g, b = (int(color[0.55][i:i + 2], 16) for i in (1, 3, 5))
    assert r > b + 40
    # The lightest blue is still clearly distinguishable from white.
    r, g, b = (int(color[-0.15][i:i + 2], 16) for i in (1, 3, 5))
    assert b - r > 20


def test_blue2red_cmap_is_discrete_with_symmetric_white():
    import numpy as np
    from matplotlib.colors import BoundaryNorm, to_hex

    from esp_lab.utils.colormap_utils import blue2red_cmap

    for levels in (np.arange(-0.5, 0.5 + 0.025, 0.05), np.arange(-4.0, 4.0 + 0.25, 0.5)):
        cmap = blue2red_cmap(levels)
        assert cmap.N == len(levels) - 1
        norm = BoundaryNorm(levels, ncolors=cmap.N)
        mids = (levels[:-1] + levels[1:]) / 2
        whites = [m for m in mids if to_hex(cmap(norm(m))) == "#ffffff"]
        assert len(whites) == 2 and np.isclose(whites[0], -whites[1])
    # An integer count of boundaries still works for symmetric ranges.
    assert blue2red_cmap(21).N == 20
