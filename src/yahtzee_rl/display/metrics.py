import matplotlib.pyplot as plt
from scipy.stats import iqr
import numpy as np
import pprint as pp
from typing import Sequence, Optional


def plot_scorecard_action_heatmap(action_counts: np.ndarray,
                                  category_labels: Sequence[str],
                                  title: str = "Scorecard Action Heatmap",
                                  show: bool = True) -> None:
    """
    Plot a heatmap of scorecard actions (turns x categories).

    Args:
        action_counts: 2D array of counts with shape (turns, categories)
        category_labels: labels for each category (x-axis)
        title: title for the plot
        show: whether to call plt.show()
    """
    if action_counts.ndim != 2:
        raise ValueError("action_counts must be a 2D array")
    if action_counts.shape[1] != len(category_labels):
        raise ValueError("action_counts columns must match category_labels length")

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    im = ax.imshow(action_counts, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel("Turn")

    ax.set_xticks(np.arange(len(category_labels)))
    ax.set_xticklabels(category_labels, rotation=45, ha="right")

    turn_labels = [str(i + 1) for i in range(action_counts.shape[0])]
    ax.set_yticks(np.arange(action_counts.shape[0]))
    ax.set_yticklabels(turn_labels)

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()

    if show:
        plt.show()


def plot_standard_metrics(X: np.ndarray, Y: np.ndarray, N: int,
                          title: str, xlabel: str = "Simulations", ylabel: str = "Score",
                          action_counts: Optional[np.ndarray] = None,
                          action_labels: Optional[Sequence[str]] = None,
                          action_title: str = "Scorecard Action Heatmap") -> None:
    """
    Plot standard metrics for a X, Y dataset, with provided N, title, xlabel, and ylabel.
    Args:
        X: The x-axis data
        Y: The y-axis data
        N: The number of data points
        title: The title of the plot
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        action_counts: optional turns x categories matrix for heatmap plotting
        action_labels: optional category labels aligned to action_counts columns
        action_title: heatmap title
    Returns:
        None
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9,9))
    Y = np.array(Y)
    mean_y = Y.mean()
    median_y = np.median(Y)
    inter_quartile_range = iqr(Y)
    sigma_y = np.std(Y)
    ax.plot(X, Y, color='b', marker='o', linestyle='dashed', linewidth=.2)
    ax.hlines(mean_y, 0, N, color='r', label=f"Mean: {mean_y}")
    ax.hlines(median_y, 0, N, color='orange', label=f"Median: {median_y}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    n, bins, patches = ax2.hist(Y, bins=50, density=True)
    Y_Norm = ((1 / (np.sqrt(2 * np.pi) * sigma_y)) *
            np.exp(-0.5 * (1 / sigma_y * (bins - mean_y)) ** 2))
    ax2.plot(bins, Y_Norm, '--')
    ax2.set_title(f"{title} Simulation Histogram")
    ax2.set_xlabel(ylabel)
    ax2.set_ylabel("Counts")
    print(f"Mean of Y: {mean_y}")
    print(f"Median of Y: {median_y}")
    pp.pprint(f"IQR of Y: {inter_quartile_range}")
    pp.pprint(f"Standard Deviation: {np.std(Y)}")

    if action_counts is not None and action_labels is not None:
        plot_scorecard_action_heatmap(
            action_counts=action_counts,
            category_labels=action_labels,
            title=action_title,
            show=False,
        )

    plt.show()