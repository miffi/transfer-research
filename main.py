import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datasets import DataLoader
from dpls.models import PLS, DeepPLS


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


def main():
    loader = DataLoader("AEEEM")
    [(name_X, (X_data, X_classes)), (name_Y, (Y_data, Y_classes))] = (
        loader.get_random_datasets()
    )

    dpls = DeepPLS(
        lv_dimensions=[20, 7],
        pls_solver="iter",
        use_nonlinear_mapping=False,
        mapping_dimensions=[],
        nys_gamma_values=[],
        stack_previous_lv1=False,
    )

    dpls.fit(X_data.T, Y_data.T)
    X_fit, Y_fit = dpls.transform(X_data.T, Y_data.T)

    X_fit = X_data @ X_fit
    Y_fit = Y_data @ Y_fit

    logi = LogisticRegression()
    logi.fit(X_fit, X_classes)
    Y_classes_pred = logi.predict(Y_fit)

    print(f"X = {name_X}, Y = {name_Y}")
    print(score(y_true=Y_classes, y_pred=Y_classes_pred))


if __name__ == "__main__":
    main()
