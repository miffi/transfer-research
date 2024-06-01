import os
import pathlib
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
from scipy.io import loadmat

DATASET_DIR = pathlib.Path("./datasets/")

DataSet: typing.TypeAlias = tuple[npt.ArrayLike, npt.ArrayLike]


LoaderFunc: typing.TypeAlias = typing.Callable[
    [os.PathLike | str], tuple[npt.ArrayLike, npt.ArrayLike]
]


def load_excel_dataset(file, class_name: str) -> DataSet:
    df = pd.read_excel(file)
    y = th.tensor(df[class_name], dtype=th.float32)
    X = th.tensor(df.drop([class_name], axis=1).to_numpy(), dtype=th.float32)
    return (X, y)


aeeem = lambda file: load_excel_dataset(file, "class")
nasa = lambda file: load_excel_dataset(file, "Defective")
relink = lambda file: load_excel_dataset(file, "isDefective")
softlab = lambda file: load_excel_dataset(file, "defects")


class DataLoader:
    _loaders: dict[str, LoaderFunc] = {
        "AEEEM": aeeem,
        "NASA": nasa,
        "RELINK": relink,
        "SOFTLAB": softlab,
    }

    def __init__(self, directory: str):
        self.directory = DATASET_DIR / directory

    def get_dataset(self, name: str) -> DataSet:
        file = next(self.directory.glob(f"{name}.*"))
        return self._load(file)

    def _load(self, file: os.PathLike | str) -> DataSet:
        return self._loaders[self.directory.name](file)

    def get_random_datasets(self, num: int = 2) -> typing.Iterable[tuple[str, DataSet]]:
        files = np.random.choice(
            np.array(list(self.directory.glob("*"))), 2, replace=False
        )
        return ((file.name, self._load(file)) for file in files)
