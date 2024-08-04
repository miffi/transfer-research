import argparse
import sys
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
from scipy.stats.distributions import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from model.datasets import DataLoader
from model.dpls.models import PLS, DeepPLS
from model.mgdpls import ModifiedGDPLS, Transfer


def score(y_true, y_pred):
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

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_fit, X_classes)

    Y_classes_pred = knn.predict(Y_fit)
    return f1_score(Y_classes, Y_classes_pred)


def exhaust(param, datas):
    size = len(datas)
    total = 0
    count = 0
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            count += 1

            (name_X, (X_data, X_classes)), (name_Y, (Y_data, Y_classes)) = (
                datas[i],
                datas[j],
            )

            gdpls = ModifiedGDPLS(
                lv_dimensions=param["lv_dimensions"],
                kernel=param["kernel"],
                pls_solver="iter",
                mapping_dimensions=param["mapping_dimensions"],
                nys_gamma_values_x=[
                    param["nys_gamma_value_x1"],
                    param["nys_gamma_value_x2"],
                ],
                nys_gamma_values_y=[
                    param["nys_gamma_value_y1"],
                    param["nys_gamma_value_y2"],
                ],
                stack_previous_lv1=True,
            )

            total += run_transfer(gdpls, X_data, Y_data, X_classes, Y_classes)
    return total / count


def print_results(datas, params: dict[str, typing.Any]):
    print(f"{params}")
    print("source\ttarget\tf1")
    for i in range(len(datas)):
        for j in range(len(datas)):
            if i == j:
                continue

            (name_X, (X_data, X_classes)), (name_Y, (Y_data, Y_classes)) = (
                datas[i],
                datas[j],
            )

            gdpls = ModifiedGDPLS(
                lv_dimensions=params["lv_dimensions"],
                kernel=params["kernel"],
                pls_solver="iter",
                mapping_dimensions=params["mapping_dimensions"],
                nys_gamma_values_x=params["nys_gamma_values_x"],
                nys_gamma_values_y=params["nys_gamma_values_y"],
                stack_previous_lv1=True,
            )

            gdpls_score = run_transfer(gdpls, X_data, Y_data, X_classes, Y_classes)
            print(f"{name_X}\t{name_Y}\t{gdpls_score:.3f}")


def hyperparameter_search(param_grid):
    print(f"Dataset: {dataset_name}, Total: {len(params)}")
    max = 0
    for i, params in enumerate(param_grid):
        print(f"\r{i} ", end="")
        val = exhaust(params, datas)
        if val > max:
            max = val
            print(max, params)
        print(f"{val}", end="")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="transfer.py",
        description="Runner for the transfer learning models.",
    )
    parser.add_argument("-s", "--seed", type=int, help="Set the random seed to a value")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    # dataset_name = "AEEEM"
    # lv_dimensions = [10, 2]
    # mapping_dimensions = [28, 35]
    # nys_gamma_values = [9.95, 9.95]
    # kernel = "rbf"

    nasa = {
        "dataset_name": "NASA",
        "lv_dimensions": [10, 2],
        "mapping_dimensions": [31, 15],
        "nys_gamma_values_x": [10.01, 8.171],
        "nys_gamma_values_y": [7.173, 0.5931],
        "kernel": "rbf",
    }

    loader = DataLoader(nasa["dataset_name"])

    datas = list(loader.get_datasets())

    # lv_dimensions = [10, 2]
    # nys_gamma_value = uniform(loc=0.001, scale=2)
    # mapping_dimensions = [(i, j) for i in range(10, 36) for j in range(10, 36)]
    #
    # params = ParameterSampler(
    #     {
    #         "mapping_dimensions": mapping_dimensions,
    #         "nys_gamma_value_x1": nys_gamma_value,
    #         "nys_gamma_value_x2": nys_gamma_value,
    #         "nys_gamma_value_y1": nys_gamma_value,
    #         "nys_gamma_value_y2": nys_gamma_value,
    #         "lv_dimensions": [lv_dimensions],
    #         "kernel": ["rbf", "laplacian"],
    #     },
    #     5000,
    # )
    # hyperparameter_search(params)

    print_results(datas, nasa)
    return 0
