"""Monochrome editorial chart style: near-black surface, off-white ink, grey ramp.
Series identity never relies on colour alone: lightness steps plus line style / marker."""
import matplotlib as mpl
import matplotlib.pyplot as plt

BG = "#0d0d0d"
INK = "#ededed"
INK2 = "#a3a3a3"
MUTED = "#6b6b6b"
GRID = "#262626"
SERIES = ["#f2f2f2", "#9e9e9e", "#5c5c5c"]     # light -> dark grey, fixed order
ACCENT = "#ffffff"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "regular",
        "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    })


def title(ax, main, sub=None):
    main = main.replace("$", r"\$")
    sub = sub.replace("$", r"\$") if sub else sub
    ax.set_title(main, color=INK, loc="left", fontsize=11.5, pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")


def note(fig, text):
    fig.text(0.01, -0.02, text, color=MUTED, fontsize=7.5, ha="left", va="top")
