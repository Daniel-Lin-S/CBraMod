import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def scatter_with_labels(
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        save_path: str='scatter_plot.png',
        label_name: str = "Label",
        xlabel: str = "X-axis",
        ylabel: str = "Y-axis",
        title: str = "Scatter Plot",
        cmap_name: str = "tab10",
        color_mode: str = "legend",
        dpi: int = 500) -> None:
    """
    Generate a scatter plot with distinct colors for unique labels.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates.
    y : np.ndarray
        1D array of y-coordinates.
    labels : np.ndarray
        1D array of labels (numbers or strings).
    save_path : str
        Path to save the generated figure.
    label_name : str
        Legend title.
    xlabel : str
        Label for X-axis.
    ylabel : str
        Label for Y-axis.
    title : str
        Plot title.
    cmap_name : str
        Matplotlib colormap name to use.
    color_mode : str
        Mode for coloring points, either 'legend' or 'colorbar'. \n
        If 'legend', each label gets a unique color.
        If 'continuous', colors are mapped to labels as a continuous spectrum.
        This requires numeric labels 
    dpi : int
        Dots per inch for saved figure.
    """
    if not (len(x) == len(y) == len(labels)):
        raise ValueError("x, y, and labels must all have the same length.")

    fig, ax = plt.subplots(figsize=(10, 8))

    if color_mode == 'legend':
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        cmap = get_cmap(cmap_name, num_classes)
        colors = cmap(np.linspace(0, 1, num_classes))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(x[mask], y[mask], color=colors[i], label=str(label), s=20)
        
    elif color_mode == 'colorbar':
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError(
                "For 'colorbar' mode, labels must be numeric."
            )
        scatter = ax.scatter(
            x, y, c=labels, cmap=cmap_name, s=20
        )
        fig.colorbar(scatter, ax=ax, label=label_name)
    else:
        raise ValueError(
            "color_mode must be either 'legend' or 'colorbar'.")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        title=label_name,bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print('Figure saved to {}'.format(save_path))
