import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.types import Metric

# The code in this file is adapted from https://stackoverflow.com/a/78122416


def radar(
    metric: Metric,
    *,
    title: str | None = None,
    max_values: dict[str, float] | None = None,
    padding: float = 1.25,
):
    df = pd.concat([metric.our, metric.others], axis=1).transpose().head(5)
    categories = df._get_numeric_data().columns.tolist()
    data = df[categories].to_dict(orient="list")
    ids = list(df.index)
    if max_values is None:
        max_values = {key: padding * max(value) for key, value in data.items()}

    normalized_data = {key: np.array(value) / 0.3 for key, value in data.items()}
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        ax.plot(angles, values, label=model_name)
        if model_name == "our":
            ax.fill(angles, values, alpha=0.15)
        for _x, _y, t in zip(angles, values, actual_values):
            t = f"{t:.2f}" if isinstance(t, float) else str(t)
            ax.text(_x, _y, t, size="xx-small")

    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks)
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    if title is not None:
        plt.suptitle(title)
    plt.show()
