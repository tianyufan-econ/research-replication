"""
figure_textsize.py
==================
Saving helpers for the multi-panel figures, which the paper draws on a smaller canvas
so their text prints at the size of the other figures (same logic as the full
package's figure_textsize.py).

    savefig_aspect(fig, filename, aspect)
        config.savefig, with the tight box padded symmetrically to the page aspect
        ratio (width / height) of the file in the paper.
    wrap_long_ylabels(fig, axes, limit=1.25)
        y-labels taller than `limit` x the axes height are split onto two lines at
        the space nearest their middle (words never change).
"""

import io

import matplotlib
from matplotlib.transforms import Bbox

from config import savefig


def savefig_aspect(fig, filename, aspect, **kwargs):
    """Save like config.savefig (dpi=300, tight box), padded to the page aspect ratio."""
    kwargs.setdefault("dpi", 300)
    rec = {}
    orig = fig.get_tightbbox

    def recording(*a, **k):                    # the tight box savefig itself computes (dry run)
        rec["bbox"] = bb = orig(*a, **k)
        return bb

    fig.get_tightbbox = recording
    try:
        fig.savefig(io.BytesIO(), format="pdf", bbox_inches="tight", **kwargs)
    finally:
        del fig.get_tightbbox
    bb = rec["bbox"].padded(matplotlib.rcParams["savefig.pad_inches"])
    x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        x0, x1 = x0 - (h * aspect - w) / 2, x1 + (h * aspect - w) / 2
    else:
        y0, y1 = y0 - (w / aspect - h) / 2, y1 + (w / aspect - h) / 2
    savefig(fig, filename, bbox_inches=Bbox([[x0, y0], [x1, y1]]), **kwargs)


def _split_middle(s):
    """Split s at the space nearest its middle (never inside $...$)."""
    if "\n" in s:
        return None
    inmath, spaces = False, []
    for i, ch in enumerate(s):
        if ch == "$":
            inmath = not inmath
        elif ch == " " and not inmath:
            spaces.append(i)
    if not spaces:
        return None
    i = min(spaces, key=lambda j: abs(j - len(s) / 2))
    return s[:i] + "\n" + s[i + 1:]


def wrap_long_ylabels(fig, axes, limit=1.25):
    """Wrap every y-label taller than limit x its axes height; returns the wrapped labels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    wrapped = []
    for ax in axes:
        t = ax.yaxis.label
        if t.get_text() and t.get_window_extent(r).height > limit * ax.get_window_extent(r).height:
            new = _split_middle(t.get_text())
            if new:
                wrapped.append(t.get_text())
                t.set_text(new)
    return wrapped
