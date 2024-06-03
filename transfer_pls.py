import torch as th
from sklearn.kernel_approximation import Nystroem

from dpls.models import PLS


class TransferGDPLS(object):
    """
    Generalized Deep PLS with transfer learning transformations

    Modified from https://github.com/kxytim/DeepPLS/blob/bf8d60d14f3fc2f5088195a8eed614eb3c5e29fb/models.py
    """

    def __init__(
        self,
        lv_dimensions: list[int],
        pls_solver: str,
        mapping_dimensions: list[int],
        nys_gamma_values: list[int],
        stack_previous_lv1: bool,
    ):
        self.lv_dimensions = lv_dimensions
        self.n_layers = len(self.lv_dimensions)
        self.pls_solver = pls_solver
        self.latent_variables_X = []
        self.latent_variables_Y = []
        self.pls_funcs = []
        self.mapping_dimensions = mapping_dimensions
        self.nys_gamma_values = nys_gamma_values
        self.mapping_funcs_X = []
        self.mapping_funcs_Y = []
        self.stack_previous_lv1 = stack_previous_lv1
        assert len(self.lv_dimensions) == len(self.mapping_dimensions)
        assert len(self.mapping_dimensions) == len(self.nys_gamma_values)

    def fit(self, X, Y):
        for layer_index in range(self.n_layers):
            nys_func_X = Nystroem(
                kernel="rbf",
                gamma=self.nys_gamma_values[layer_index],
                n_components=self.mapping_dimensions[layer_index],
                n_jobs=-1,
            )

            nys_func_Y = Nystroem(
                kernel="rbf",
                gamma=self.nys_gamma_values[layer_index],
                n_components=self.mapping_dimensions[layer_index],
                n_jobs=-1,
            )

            X_backup = X.clone()
            X = nys_func_X.fit_transform(X)
            self.mapping_funcs_X.append(nys_func_X)
            X = th.tensor(X)
            if self.stack_previous_lv1 and layer_index > 0:
                lv1_previous_layer = X_backup[:, [0]]
                X = th.hstack((lv1_previous_layer, X))

            Y_backup = Y.clone()
            Y = nys_func_Y.fit_transform(Y)
            self.mapping_funcs_Y.append(nys_func_Y)
            Y = th.tensor(Y)
            if self.stack_previous_lv1 and layer_index > 0:
                lv1_previous_layer = Y_backup[:, [0]]
                Y = th.hstack((lv1_previous_layer, Y))

            pls = PLS(
                n_components=self.lv_dimensions[layer_index], solver=self.pls_solver
            )
            pls.fit(X, Y)
            self.pls_funcs.append(pls)

            latent_variables_X = pls.x_scores_
            self.latent_variables_X.append(latent_variables_X)
            X = latent_variables_X

            latent_variables_Y = pls.y_scores_
            self.latent_variables_Y.append(latent_variables_Y)
            Y = latent_variables_Y
        return self

    def transform(self, test_X, test_Y=None):
        for layer_index in range(self.n_layers):
            test_X_backup = test_X.clone()
            test_X = self.mapping_funcs_X[layer_index].transform(test_X)
            test_X = th.tensor(test_X)
            if self.stack_previous_lv1 and layer_index > 0:
                lv1_previous_layer = test_X_backup[:, [0]]
                test_X = th.hstack((lv1_previous_layer, test_X))

            test_Y_backup = test_Y.clone()
            test_Y = self.mapping_funcs_Y[layer_index].transform(test_Y)
            test_Y = th.tensor(test_Y)
            if self.stack_previous_lv1 and layer_index > 0:
                lv1_previous_layer = test_Y_backup[:, [0]]
                test_Y = th.hstack((lv1_previous_layer, test_Y))

            if layer_index + 1 == self.n_layers:
                return test_X, test_Y
            test_X, test_Y = self.pls_funcs[layer_index].transform(test_X, test_Y)
