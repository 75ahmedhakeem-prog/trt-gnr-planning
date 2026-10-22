"""Common figure style (Origin-like, large fonts, no shaded ranges)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 16.5, "font.weight": "bold", "axes.labelweight": "bold",
    "axes.labelsize": 18, "axes.linewidth": 1.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "xtick.major.size": 7, "ytick.major.size": 7, "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5, "legend.frameon": False, "legend.fontsize": 13.5,
    "lines.linewidth": 2.2, "savefig.dpi": 600, "savefig.bbox": "tight"})
BLUE, RED, GREEN, ORANGE, PURPLE, BLACK, GREY = "#1F4FB4", "#D62728", "#2CA02C", "#E67E22", "#6A3D9A", "#000000", "#7F7F7F"
def lab(ax, s, x=0.04, y=0.95, ha="left", **k): ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=19, fontweight="bold", **k)
