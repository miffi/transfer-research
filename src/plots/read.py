import os
from pathlib import Path

import pandas as pd

from plots.types import Metric

# files in each directory containing our and others' data respectively
OUR_FILE = "our.tsv"
OTHERS_FILE = "others.tsv"

RESOURCES_DIRECTORY = Path("resources/results/")


def _read(file: str | os.PathLike) -> pd.DataFrame:
    return pd.read_csv(file, index_col=[0, 1], sep="\t")


def _check_same_indices(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    # adapted from https://stackoverflow.com/questions/59198707/determining-if-several-data-frames-have-the-same-column-names-in-pandas
    return len(df1.index.intersection(df2.index)) == df1.shape[0]


def load_metrics(dataset: str | os.PathLike, metric: str | os.PathLike) -> Metric:
    directory = RESOURCES_DIRECTORY / dataset / metric
    others = _read(directory / OTHERS_FILE)
    our = _read(directory / OUR_FILE)
    assert _check_same_indices(our, others)
    return Metric(dataset_name=str(dataset), others=others, our=our)
