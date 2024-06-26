import argparse

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from datasets import DataLoader
from dpls.models import PLS, DeepPLS
from transfer_pls import TransferGDPLS


def score(
    y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]
) -> dict[str, float]:
    """
    Calculates some scores for results of predictions. `y_true` and `y_pred` need to have the same dimensions.
    :param y_true: The true value of the tests.
    :param y_pred: The predicted value of the tests.

    :returns A dict with the keys "f1" and "roc_auc" with their values set to the corresponding calculations between the parameters.
    """
    return {
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
    }


class Transfer:
    """Transfer learning for a given model."""

    def __init__(self, model):
        """
        :param model: The model to transfer learning with.
        """
        self.model = model

    def fit(self, X, Y):
        self.model.fit(X.T, Y.T)
        return self

    def transform(self, X, Y):
        X_fit, Y_fit = self.model.transform(X.T, Y.T)

        X_fit = X @ X_fit
        Y_fit = Y @ Y_fit

        return X_fit, Y_fit

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X, Y)


def run_transfer(model, X_data, Y_data, X_classes, Y_classes) -> dict[str, float]:
    """
    Driver function to run various transfer models
    :param model: The model to transfer learning with.
    :param X_data: The training data for the source; shape (m, n)
    :param Y_data: The training data for the target; shape (k, n)
    :param X_classes: The true values for the source; shape (m, 1)
    :param Y_classes: The true values for the source; shape (k, 1)

    :return Various scores of the model in a dictionary.
    """

    transfer = Transfer(model)
    X_fit, Y_fit = transfer.fit_transform(X_data, Y_data)

    num_classes = len(np.unique(X_classes))

    knn = KNeighborsClassifier(n_neighbors=num_classes)
    knn.fit(X_fit, X_classes)

    Y_classes_pred = knn.predict(Y_fit)
    return score(y_true=Y_classes, y_pred=Y_classes_pred)


def main():
    parser = argparse.ArgumentParser(
        prog="transfer.py",
        description="Runner for the transfer learning models.",
    )
    parser.add_argument("-s", "--seed", type=int, help="Set the random seed to a value")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    loader = DataLoader("AEEEM")

    datas = list(loader.get_datasets())

    lv_dimensions = [30, 40]
    nys_gamma_values = [0.01, 0.11]
    mapping_dimensions = [50, 50]
    for i in range(len(datas)):
        for j in range(len(datas)):
            if i == j:
                continue

            (name_X, (X_data, X_classes)), (name_Y, (Y_data, Y_classes)) = (
                datas[i],
                datas[j],
            )

            dpls = DeepPLS(
                lv_dimensions=lv_dimensions,
                pls_solver="iter",
                use_nonlinear_mapping=False,
                mapping_dimensions=[],
                nys_gamma_values=[],
                stack_previous_lv1=False,
            )
            pls = PLS(n_components=lv_dimensions[-1], solver="iter")
            gdpls = TransferGDPLS(
                lv_dimensions=lv_dimensions,
                pls_solver="iter",
                mapping_dimensions=mapping_dimensions,
                nys_gamma_values=nys_gamma_values,
                stack_previous_lv1=True,
            )

            print(f"X = {name_X}, Y = {name_Y}")
            print(f"PLS  = {run_transfer(pls, X_data, Y_data, X_classes, Y_classes)}")
            print(f"DPLS = {run_transfer(dpls, X_data, Y_data, X_classes, Y_classes)}")
            print(
                f"GDPLS = {run_transfer(gdpls, X_data, Y_data, X_classes, Y_classes)}"
            )


if __name__ == "__main__":
    main()
