from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple, Literal, Optional
from scipy.stats import (
    pearsonr,
    spearmanr,
    ttest_ind,
    chi2_contingency,
)

from shared_vars import (
    project_dir,
    preprocessed_dataset,
    hip_features,
    knee_features,
    hip_features_with_spaces_to_underscores,
    knee_features_with_spaces_to_underscores,
    abridged_names,
)
import itertools
from typing import List, Tuple, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl

mpl.rcParams.update(
    {
        "xtick.major.pad": 0,
        "ytick.major.pad": 0,
    }
)


preprocessed_dataset_renamed = preprocessed_dataset.rename(
    columns={
        feature: abridged_names.get(feature, feature)
        for feature in preprocessed_dataset.columns
    }
)


def plot_pairs(
    df: pd.DataFrame,
    vars: Optional[List[str]] = None,
    *,
    htest: bool = True,
    figsize: Tuple[float, float] = (7.0, 7.0),
    max_vars: int = 10,
    corr_method: Literal["pearson", "spearman"] = "pearson",
    scatter_alpha: float = 0.6,
    hist_kde: bool = True,
) -> plt.Figure:
    if vars is None:
        vars = list(df.columns)
    if len(vars) > max_vars:
        raise ValueError(
            f"Plotting {len(vars)} variables is unwieldy; try ≤ {max_vars}."
        )
    data = df[vars].dropna()
    n = len(vars)
    _corr = pearsonr if corr_method == "pearson" else spearmanr

    fig, axes = plt.subplots(
        n,
        n,
        figsize=figsize,
    )
    if n == 1:
        axes = np.array([[axes]])

    def _diag(ax: plt.Axes, s: pd.Series, flip: bool, xticks: bool) -> None:
        if not flip:
            if is_numeric_dtype(s):
                sns.histplot(
                    x=s,
                    kde=hist_kde,
                    stat="density",
                    ax=ax,
                    edgecolor="white",
                )
            else:
                sns.countplot(
                    x=s.astype(str), order=sorted(s.astype(str).unique()), ax=ax
                )
            if not xticks:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.tick_params(axis="x", labelrotation=0, labelsize=5)
                ax.set_yticks([])
        else:
            if is_numeric_dtype(s):
                sns.histplot(
                    y=s,
                    kde=hist_kde,
                    stat="density",
                    ax=ax,
                    edgecolor="white",
                )
            else:
                sns.countplot(
                    y=s.astype(str), order=sorted(s.astype(str).unique()), ax=ax
                )
            if xticks:
                ax.tick_params(axis="y", labelrotation=0, labelsize=5)

    def _scatter_or_box(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
        num_x, num_y = is_numeric_dtype(x), is_numeric_dtype(y)
        if num_x and num_y:
            sns.scatterplot(x=x, y=y, ax=ax, alpha=scatter_alpha, edgecolor="none", s=1)
            ax.tick_params(axis="x", labelrotation=0, labelsize=5)
            ax.tick_params(axis="y", labelrotation=0, labelsize=5)
        elif num_x and not num_y:
            sns.boxplot(x=y.astype(str), y=x, ax=ax)
            ax.tick_params(axis="x", labelrotation=0, labelsize=5)
            ax.tick_params(axis="y", labelrotation=0, labelsize=5)
        elif not num_x and num_y:
            sns.boxplot(
                x=x.astype(str),
                y=y,
                ax=ax,
                linewidth=1,
                flierprops=dict(marker="o", markersize=1),
            )
            ax.tick_params(axis="x", labelrotation=0, labelsize=5)
            ax.tick_params(axis="y", labelrotation=0, labelsize=5)
        else:
            ct = pd.crosstab(x.astype(str), y.astype(str)).T
            sns.heatmap(
                ct,
                cmap="Blues",
                annot=True,
                fmt="d",
                cbar=False,
                ax=ax,
                annot_kws={"fontsize": 7},
            )
            ax.tick_params(axis="x", labelrotation=0, labelsize=5)
            ax.tick_params(axis="y", labelrotation=0, labelsize=5)
            ax.set_xlabel("")
            ax.set_ylabel("")

    def _annot_corr(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
        num_x, num_y = is_numeric_dtype(x), is_numeric_dtype(y)
        if num_x and num_y:
            r, p = _corr(x, y)
            txt = f"{r:+.2f}\n(p={p:.2f})" if htest else f"{r:+.3f}"
        elif num_x ^ num_y:
            num = x if num_x else y
            cat = y if num_x else x
            groups = cat.dropna().unique()

            if len(groups) == 2:
                g1, g2 = (num[cat == g] for g in groups)
                stat, p = ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
                txt = f"t={stat:+.2f}\n(p={p:.2f})" if htest else f"t={stat:+.2f}"
            else:
                txt = "n>2"
        else:
            ct = pd.crosstab(x, y)
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, p, _, _ = chi2_contingency(ct)
                txt = f"χ²={chi2:.1f}\n(p={p:.2f})" if htest else f"χ²={chi2:.1f}"
            else:
                txt = "–"
        ax.text(
            0.5, 0.5, txt, ha="center", va="center", transform=ax.transAxes, fontsize=7
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for i, y_var in enumerate(vars):
        for j, x_var in enumerate(vars):
            ax = axes[i, j]
            ax.grid(False)
            if i == j:
                _diag(
                    ax,
                    data[x_var],
                    flip=(i < len(vars) / 2),
                    xticks=(j == 0 or i == n - 1),
                )
            elif i > j:
                _scatter_or_box(ax, data[x_var], data[y_var])
            else:
                _annot_corr(ax, data[x_var], data[y_var])
            if i == n - 1:
                ax.set_xlabel(x_var, rotation=45, ha="right", fontsize=6)
            else:
                ax.set_xlabel("")
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(y_var, rotation=45, ha="right", fontsize=6)
            else:
                ax.set_ylabel("")
                ax.set_yticks([])

    fig.tight_layout(
        pad=0.2,
        w_pad=0.2,
        h_pad=0.2,
    )
    return fig


knee_features_abridged = [
    abridged_names.get(feature, feature) for feature in knee_features
]


fig = plot_pairs(
    df=preprocessed_dataset_renamed,
    vars=["Sex", "Varus vs Valgus", "Weight", "Age"]
    + hip_features
    + knee_features_abridged,
    htest=True,
    figsize=(15, 15),
    max_vars=20,
    corr_method="pearson",
    scatter_alpha=0.6,
    hist_kde=True,
)
fig.savefig(
    project_dir / "results" / "figures" / "pairplot_hip_knee_features.png",
    dpi=300,
    bbox_inches="tight",
)
