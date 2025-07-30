from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from shared_vars import (
    project_dir,
    preprocessed_dataset,
    hip_features,
    knee_features,
    abridged_names,
)


sns.set_theme(context="paper", style="whitegrid")


def plot_distributions_stratifying_by_varus_and_sex(
    features: list[str],
    dataset: pd.DataFrame,
    subtitle_font_size: int = 12,
    xylabel_font_size: int = 10,
    tick_font_size: int = 8,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=len(features), ncols=3, figsize=figsize)
    for i, feature in enumerate(features):
        axes_row = axes[i]

        # first, plot the distribution of the feature
        sns.histplot(dataset[feature], kde=True, ax=axes_row[0], color="blue", bins=30)
        axes_row[0].set_title(f"Distribution of {feature}", fontsize=subtitle_font_size)
        axes_row[0].set_xlabel(feature, fontsize=xylabel_font_size)
        axes_row[0].set_ylabel("Frequency", fontsize=xylabel_font_size)

        # then, plot the distribution of the feature stratified by Varus Alignment
        sns.histplot(
            dataset[dataset["Varus vs Valgus"] == 1][feature],
            kde=True,
            ax=axes_row[1],
            color="green",
            bins=30,
            label="Varus",
        )
        sns.histplot(
            dataset[dataset["Varus vs Valgus"] == 0][feature],
            kde=True,
            ax=axes_row[1],
            color="red",
            bins=30,
            label="Valgus",
        )
        axes_row[1].set_title(
            f"{feature} by Frontal Plane Alignment", fontsize=subtitle_font_size
        )
        axes_row[1].set_xlabel(feature, fontsize=xylabel_font_size)
        axes_row[1].set_ylabel("Frequency", fontsize=xylabel_font_size)
        axes_row[1].legend()
        axes_row[1].tick_params(labelsize=tick_font_size)

        # finally, plot the distribution of the feature stratified by Sex
        sns.histplot(
            dataset[dataset["Sex"] == "F"][feature],
            kde=True,
            ax=axes_row[2],
            color="green",
            bins=30,
            label="Female",
        )
        sns.histplot(
            dataset[dataset["Sex"] == "M"][feature],
            kde=True,
            ax=axes_row[2],
            color="red",
            bins=30,
            label="Male",
        )
        axes_row[2].set_title(f"{feature} by Sex", fontsize=subtitle_font_size)
        axes_row[2].set_xlabel(feature, fontsize=xylabel_font_size)
        axes_row[2].set_ylabel("Frequency", fontsize=xylabel_font_size)
        axes_row[2].tick_params(labelsize=tick_font_size)
        axes_row[2].legend()

    plt.tight_layout()
    return fig


def plot_distributions_grid(
    features: list[str],
    dataset: pd.DataFrame,
    xsubtitle_font_size: int = 12,
    ysubtitle_font_size: int = 12,
    tick_font_size: int = 8,
    figsize: tuple[int, int] = (12, 2.2),
) -> plt.Figure:
    """
    Grid of histograms (rows = features, columns = plot type).
    No axes are shared, so every subplot keeps its own x-scale.
    """
    n_rows, n_cols = len(features), 3
    col_titles = ["Overall Distribution", "Varus vs Valgus", "Female vs Male"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    axes = np.atleast_2d(axes)

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=xsubtitle_font_size)

    for i, feature in enumerate(features):
        if feature in abridged_names:
            feature_name = abridged_names[feature]
        else:
            feature_name = feature
        axes[i, 0].set_ylabel(
            feature_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=ysubtitle_font_size,
            labelpad=10,
        )
        sns.histplot(
            dataset[feature], kde=True, bins=30, color="steelblue", ax=axes[i, 0]
        )
        sns.histplot(
            dataset.loc[dataset["Varus vs Valgus"] == "Varus", feature],
            kde=True,
            bins=30,
            color="forestgreen",
            ax=axes[i, 1],
            label="Varus",
            alpha=0.6,
        )
        sns.histplot(
            dataset.loc[dataset["Varus vs Valgus"] == "Valgus", feature],
            kde=True,
            bins=30,
            color="darkorange",
            ax=axes[i, 1],
            label="Valgus",
            alpha=0.6,
        )
        if i == 0:
            axes[i, 1].legend(frameon=False, fontsize=8)

        sns.histplot(
            dataset.loc[dataset["Sex"] == "F", feature],
            kde=True,
            bins=30,
            color="firebrick",
            ax=axes[i, 2],
            label="Female",
            alpha=0.6,
        )
        sns.histplot(
            dataset.loc[dataset["Sex"] == "M", feature],
            kde=True,
            bins=30,
            color="royalblue",
            ax=axes[i, 2],
            label="Male",
            alpha=0.6,
        )
        if i == 0:
            axes[i, 2].legend(frameon=False, fontsize=8)
        for j in range(n_cols):
            axes[i, j].set_xlabel("")
            axes[i, j].tick_params(labelsize=tick_font_size)
            axes[i, j].grid(False)
            if j > 0:
                axes[i, j].set_ylabel("")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    hip_features_fig = plot_distributions_grid(
        features=hip_features,
        dataset=preprocessed_dataset,
        xsubtitle_font_size=12,
        ysubtitle_font_size=12,
        tick_font_size=8,
        figsize=(8, 2),
    )
    hip_features_fig.savefig(
        project_dir / "results" / "figures" / "hip_features_distribution.png", dpi=300
    )

    knee_features_fig = plot_distributions_grid(
        features=knee_features,
        dataset=preprocessed_dataset,
        xsubtitle_font_size=12,
        ysubtitle_font_size=12,
        tick_font_size=8,
        figsize=(8, 2),
    )
    knee_features_fig.savefig(
        project_dir / "results" / "figures" / "knee_features_distribution.png", dpi=300
    )

    other_features_fig = plot_distributions_grid(
        features=["Age", "Weight"],
        dataset=preprocessed_dataset,
        xsubtitle_font_size=12,
        ysubtitle_font_size=12,
        tick_font_size=8,
        figsize=(8, 2),
    )
    other_features_fig.savefig(
        project_dir / "results" / "figures" / "other_features_distribution.png", dpi=300
    )
