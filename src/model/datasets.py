import os
import pathlib
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
from scipy.io import loadmat

DATASET_DIR = pathlib.Path("./resources/datasets/")

DataSet: typing.TypeAlias = tuple[npt.ArrayLike, npt.ArrayLike]


LoaderFunc: typing.TypeAlias = typing.Callable[
    [os.PathLike | str], tuple[npt.ArrayLike, npt.ArrayLike]
]


def _load_excel_dataset(file, class_name: str) -> DataSet:
    df = pd.read_excel(file)
    y = th.tensor(df[class_name], dtype=th.float32)
    X = th.tensor(df.drop([class_name], axis=1).to_numpy(), dtype=th.float32)
    return (X, y)


_aeeem = lambda file: _load_excel_dataset(file, "class")
_nasa = lambda file: _load_excel_dataset(file, "Defective")
_relink = lambda file: _load_excel_dataset(file, "isDefective")
_softlab = lambda file: _load_excel_dataset(file, "defects")


def _load_matlab(file):
    data = loadmat(file)
    print(data)
    print(
        data["X_src"].shape,
        data["X_tar"].shape,
        data["Y_src"].shape,
        data["Y_tar"].shape,
    )
    exit()


def _caltech_and_amazon(file):
    data = loadmat(file)
    return (
        th.tensor(data["fts"], dtype=th.float32),
        th.tensor(data["labels"], dtype=th.float32).ravel(),
    )


def _image(file):
    data = loadmat(file)

    X = np.vstack([data["X_src"].T, data["X_tar"].T])
    y = np.vstack([data["Y_src"], data["Y_tar"]]).ravel()
    return th.tensor(X, dtype=th.float32), th.tensor(y, dtype=th.float32)


def _text(file):
    data = loadmat(file)

    print(data["Xs"].shape, data["Xt"].shape, data["Ys"].shape, data["Yt"].shape)

    X = np.vstack([data["Xs"].todense().T, data["Xt"].todense().T])
    y = np.vstack([data["Ys"].toarray(), data["Yt"].toarray()]).ravel()
    print(X.shape, y.shape)
    return th.tensor(X, dtype=th.float32), th.tensor(y, dtype=th.float32)


class DataLoader:
    _loaders: dict[str, LoaderFunc] = {
        "20news_sum": _text,
        "AEEEM": _aeeem,
        "COL20": _image,
        "NASA": _nasa,
        "OfficeCaltech": _caltech_and_amazon,
        "RELINK": _relink,
        "Reuters": _text,
        "SOFTLAB": _softlab,
        "amazon_review_400": _caltech_and_amazon,
        "mnist_usps": _image,
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

    def get_datasets(self) -> typing.Iterable[tuple[str, DataSet]]:
        return (
            (os.path.splitext(file.name)[0], self._load(file))
            for file in np.array(list(self.directory.glob("*")))
        )
