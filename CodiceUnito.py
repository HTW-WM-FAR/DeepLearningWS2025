import os
import random
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import xgboost as xgb
import optuna

import tensorflow as tf
import keras
import keras.backend as K
from keras.optimizers import AdamW

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

DATA_PATH = r"PATH OF EXCEL FILE WITH PRPOPERTIES.xlsx"
PLOT_DIR = r"PATH TO SAVE PLOTS"
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET_COLUMNS = ["daysOnMarket", "price"]


def set_global_seeds(seed: int = RANDOM_STATE) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Base seed used for Python, NumPy and TensorFlow.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# DATA LOADING AND STANDARDIZATION
# ============================================================

def load_and_standardize(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset from Excel, standardize all numeric columns,
    and return features X and targets y.

    Parameters
    ----------
    path : str
        Path to the Excel file containing the dataset.

    Returns
    -------
    X : np.ndarray
        Matrix of input features.
    y : np.ndarray
        Matrix of targets with shape (n_samples, 2)
        where columns correspond to daysOnMarket and price.
    """
    data = pd.read_excel(path)

    # Standardize all numeric columns to zero mean and unit variance
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # Features: all columns except the targets and removedDate
    X = data.drop(columns=["daysOnMarket", "price", "removedDate"], errors="ignore").values

    # Targets: daysOnMarket and price
    y = data[TARGET_COLUMNS].values

    return X, y


# ============================================================
# METRICS AND GENERIC K-FOLD EVALUATION
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute R², RMSE and MAE for both targets (DOM and price).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values, shape (n_samples, 2).
    y_pred : np.ndarray
        Predicted values, shape (n_samples, 2).

    Returns
    -------
    metrics : dict
        Dictionary containing r2_dom, rmse_dom, mae_dom,
        r2_price, rmse_price, mae_price.
    """
    metrics: Dict[str, float] = {}

    for i, name in enumerate(["dom", "price"]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        metrics[f"r2_{name}"] = r2_score(yt, yp)

        # MSE without 'squared' argument (for compatibility),
        # then take the square root to obtain RMSE
        mse = mean_squared_error(yt, yp)
        metrics[f"rmse_{name}"] = float(mse ** 0.5)

        metrics[f"mae_{name}"] = mean_absolute_error(yt, yp)

    return metrics


def kfold_evaluate(model_builder, X: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS) -> Dict[str, float]:
    """
    Evaluate a model using K-fold cross validation.

    Parameters
    ----------
    model_builder : callable
        Function that receives (X_train, y_train) and returns a fitted model
        with a .predict(X_test) method.
    X : np.ndarray
        Input features.
    y : np.ndarray
        Targets.
    n_splits : int
        Number of folds for KFold cross validation.

    Returns
    -------
    avg_metrics : dict
        Average metrics across all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    all_metrics = []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_builder(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_metrics = compute_metrics(y_test, y_pred)
        all_metrics.append(fold_metrics)

    avg_metrics = {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in all_metrics[0].keys()
    }
    return avg_metrics


# ============================================================
# PARETO FRONT AND PLOTTING UTILITIES
# ============================================================

def extract_rmse_points_from_trials(trials) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract RMSE pairs (DOM, price) from Optuna trials.

    Parameters
    ----------
    trials : list of optuna.trial.FrozenTrial
        List of completed trials.

    Returns
    -------
    xs : np.ndarray
        RMSE values for daysOnMarket.
    ys : np.ndarray
        RMSE values for price.
    """
    xs, ys = [], []
    for tr in trials:
        if tr.state != optuna.trial.TrialState.COMPLETE:
            continue
        m = tr.user_attrs.get("metrics")
        if m is None:
            continue
        xs.append(m["rmse_dom"])
        ys.append(m["rmse_price"])
    return np.array(xs), np.array(ys)


def pareto_front(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Pareto front for a set of 2D points.

    Parameters
    ----------
    xs : np.ndarray
        First objective values (e.g., RMSE DOM).
    ys : np.ndarray
        Second objective values (e.g., RMSE price).

    Returns
    -------
    pf_x : np.ndarray
        x-coordinates of Pareto-optimal points.
    pf_y : np.ndarray
        y-coordinates of Pareto-optimal points.
    """
    if len(xs) == 0:
        return xs, ys
    pts = np.column_stack([xs, ys])
    mask = paretoset(pts, sense=["min", "min"])
    pf = pts[mask]
    return pf[:, 0], pf[:, 1]


def plot_pareto(xs_soo, ys_soo, xs_moo, ys_moo, title: str, filename: str) -> None:
    """
    Plot and save Pareto front scatterplots comparing
    SOO and MOO optimization results.

    Parameters
    ----------
    xs_soo, ys_soo : array-like
        RMSE points from single-objective optimization.
    xs_moo, ys_moo : array-like
        RMSE points from multi-objective optimization.
    title : str
        Plot title.
    filename : str
        Name of the PNG file within PLOT_DIR.
    """
    plt.figure()
    plt.scatter(xs_soo, ys_soo, label="Non-dominated points with SOO", alpha=0.8)
    plt.scatter(xs_moo, ys_moo, label="Non-dominated points with MOO", alpha=0.8)
    plt.xlabel("Days On Market error (RMSE)")
    plt.ylabel("Price error (RMSE)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    full_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"Saved Pareto plot: {full_path}")


# ============================================================
# LINEAR REGRESSION BASELINE
# ============================================================

def build_linear_regression(X_train, y_train):
    """
    Build and fit a multi-output Linear Regression baseline.
    """
    base = LinearRegression()
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model


def run_linear_regression(X, y) -> Dict[str, float]:
    """
    Run Linear Regression with K-fold cross validation and print metrics.
    """
    print("=== Linear Regression (Standardized data) ===")
    metrics = kfold_evaluate(build_linear_regression, X, y)
    print(metrics)
    return metrics


# ============================================================
# XGBOOST MODELS
# ============================================================

def build_xgb(config: Dict, X_train, y_train):
    """
    Build and fit a multi-output XGBoost regressor using the given config.
    """
    base = xgb.XGBRegressor(**config)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model


def xgb_default_config() -> Dict:
    """
    Default hyperparameter configuration for XGBoost.
    """
    return {
        "objective": "reg:squarederror",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "verbosity": 0,
        "random_state": RANDOM_STATE,
    }


def run_xgb_default(X, y) -> Dict[str, float]:
    """
    Evaluate the default XGBoost configuration with K-fold cross validation.
    """
    print("=== XGBoost (Default hyperparameters) ===")
    config = xgb_default_config()
    metrics = kfold_evaluate(lambda Xtr, Ytr: build_xgb(config, Xtr, Ytr), X, y)
    print(metrics)
    return metrics


# --------- XGBoost + Optuna SOO ----------------------------------------------

def xgb_optuna_objective(trial: optuna.Trial, X, y) -> float:
    """
    Optuna objective for XGBoost single-objective optimization.
    The objective minimizes the average RMSE over both targets.
    """
    config = {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "random_state": RANDOM_STATE,
    }

    def builder(Xtr, Ytr):
        return build_xgb(config, Xtr, Ytr)

    metrics = kfold_evaluate(builder, X, y)
    # ***** CHANGED: optimize RMSE instead of MAE *****
    obj = (metrics["rmse_dom"] + metrics["rmse_price"]) / 2.0
    trial.set_user_attr("metrics", metrics)
    return obj


def run_xgb_optuna_soo(X, y, n_trials: int = 30):
    """
    Run single-objective Bayesian optimization for XGBoost using Optuna.

    Returns
    -------
    metrics : dict
        Metrics for the best configuration.
    best_config : dict
        Corresponding hyperparameters.
    pareto_points : (np.ndarray, np.ndarray)
        Pareto front points (RMSE DOM, RMSE price) extracted from all trials.
    """
    print("=== XGBoost + Optuna (SOO) ===")
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="xgb_soo")
    study.optimize(lambda t: xgb_optuna_objective(t, X, y), n_trials=n_trials)

    best_trial = study.best_trial
    best_config = best_trial.params
    best_config.update({
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "verbosity": 0,
        "random_state": RANDOM_STATE,
    })

    metrics = kfold_evaluate(lambda Xtr, Ytr: build_xgb(best_config, Xtr, Ytr), X, y)
    print("Best XGB params (SOO):", best_config)
    print("XGB SOO metrics:", metrics)

    xs, ys = extract_rmse_points_from_trials(study.trials)
    xs_pf, ys_pf = pareto_front(xs, ys)

    return metrics, best_config, (xs_pf, ys_pf)


# --------- XGBoost + Optuna MOO (weighted RMSE) ------------------------------

def run_xgb_optuna_moo_weighted(X, y, alphas=None, n_trials_per_alpha: int = 15):
    """
    Approximate multi-objective optimization for XGBoost
    by using a weighted sum of RMSE_dom and RMSE_price
    for several alpha values.

    Parameters
    ----------
    alphas : list of float
        Trade-off parameters between the two objectives.
    n_trials_per_alpha : int
        Number of Optuna trials per alpha value.

    Returns
    -------
    best_metrics_table : dict
        Metrics of the best solution (in terms of RMSE_dom + RMSE_price).
    pareto_points : (np.ndarray, np.ndarray)
        Approximated Pareto front points.
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=RANDOM_STATE)
    xs_all, ys_all = [], []
    metrics_per_alpha: Dict[float, Dict[str, float]] = {}

    for alpha in alphas:
        # Nonlinear weighting to emphasize extreme trade-offs
        w1 = alpha ** 2
        w2 = 1.0 - w1

        def objective(trial: optuna.Trial) -> float:
            config = {
                "objective": "reg:squarederror",
                "n_estimators": 1000,
                "verbosity": 0,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "random_state": RANDOM_STATE,
            }

            def builder(Xtr, Ytr):
                return build_xgb(config, Xtr, Ytr)

            metrics = kfold_evaluate(builder, X, y)
            trial.set_user_attr("metrics", metrics)
            obj = w1 * metrics["rmse_dom"] + w2 * metrics["rmse_price"]
            return obj

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials_per_alpha)

        best_trial = study.best_trial
        m = best_trial.user_attrs["metrics"]
        metrics_per_alpha[alpha] = m
        xs_all.append(m["rmse_dom"])
        ys_all.append(m["rmse_price"])

    xs_all = np.array(xs_all)
    ys_all = np.array(ys_all)
    xs_pf, ys_pf = pareto_front(xs_all, ys_all)

    # Select the overall best trade-off according to RMSE_dom + RMSE_price
    idx_best = np.argmin(xs_all + ys_all)
    best_metrics_table = list(metrics_per_alpha.values())[idx_best]

    return best_metrics_table, (xs_pf, ys_pf)


# ============================================================
# RESNET FOR TABULAR DATA
# ============================================================

def make_normalization(normalization: str, input_dim: int):
    """
    Build the requested normalization layer with the correct input shape.
    """
    if normalization == "batchnorm":
        return keras.layers.BatchNormalization(input_shape=(input_dim,))
    elif normalization == "layernorm":
        return keras.layers.LayerNormalization(input_shape=(input_dim,))
    else:
        raise ValueError("Unknown normalization: " + normalization)


class ResNetBlock(keras.layers.Layer):
    """
    Simple residual block for fully connected ResNet-style architecture.

    The block applies:
    Norm -> Dense(hidden_factor * d) + ReLU + Dropout ->
    Dense(d) + Dropout, and adds the input as a residual connection.
    """

    def __init__(
        self,
        input_dim: int,
        normalization: str,
        hidden_factor: float = 2.0,
        hidden_dropout: float = 0.1,
        residual_dropout: float = 0.05,
    ):
        super().__init__()
        d_hidden = int(hidden_factor * input_dim)
        self.ff = keras.models.Sequential([
            make_normalization(normalization, input_dim),
            keras.layers.Dense(d_hidden, activation="relu"),
            keras.layers.Dropout(hidden_dropout),
            keras.layers.Dense(input_dim),
            keras.layers.Dropout(residual_dropout),
        ])

    def call(self, x, *args, **kwargs):
        return x + self.ff(x)


def construct_resnet_model(params: Dict, input_dim: int) -> keras.Model:
    """
    Build a fully connected ResNet-like model for tabular regression.

    Parameters
    ----------
    params : dict
        Hyperparameters for the architecture and optimizer.
    input_dim : int
        Number of input features.

    Returns
    -------
    model : keras.Model
        Compiled Keras model.
    """
    n_hidden = params.get("n_hidden", 2)
    layer_size = params.get("layer_size", 64)
    normalization = params.get("normalization", "layernorm")
    hidden_factor = params.get("hidden_factor", 2.0)
    hidden_dropout = params.get("hidden_dropout", 0.1)
    residual_dropout = params.get("residual_dropout", 0.05)

    model = keras.models.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(layer_size),
    ])

    # Stack residual blocks
    for _ in range(n_hidden):
        model.add(ResNetBlock(layer_size, normalization, hidden_factor, hidden_dropout, residual_dropout))

    # Final normalization, activation and output layer
    model.add(keras.Sequential([
        make_normalization(normalization, layer_size),
        keras.layers.ReLU(),
        keras.layers.Dense(2),
    ]))

    # ***** CHANGED: use MSE loss (we'll plot RMSE from it) *****
    model.compile(
        optimizer=AdamW(
            learning_rate=params.get("learning_rate", 1e-3),
            weight_decay=params.get("weight_decay", 1e-6),
        ),
        loss="mean_squared_error",
    )
    return model


def build_and_fit_resnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict,
    verbose: bool = False,
    seed: Optional[int] = None,
):
    """
    Build and train a ResNet model on a single training set,
    using a hold-out validation split and early stopping.
    """
    if seed is not None:
        tf.random.set_seed(seed)
    model = construct_resnet_model(params, X_train.shape[-1])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=15,
        restore_best_weights=True,
        verbose=verbose,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        patience=5,
        factor=0.5,
        min_lr=1e-5,
        verbose=verbose,
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed
    )

    model.fit(
        X_tr,
        y_tr,
        batch_size=params.get("batch_size", 256),
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=verbose,
    )
    return model


def kfold_evaluate_resnet(params: Dict, X: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS) -> Dict[str, float]:
    """
    Evaluate the ResNet model with K-fold cross validation.

    The model is re-built and trained from scratch in each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        K.clear_session()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = build_and_fit_resnet(X_train, y_train, params, verbose=False, seed=fold)
        y_pred = model.predict(X_test, verbose=False)
        all_metrics.append(compute_metrics(y_test, y_pred))

    avg_metrics = {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in all_metrics[0].keys()
    }
    return avg_metrics


def resnet_default_params() -> Dict:
    """
    Default hyperparameters for the ResNet architecture.
    """
    return {
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "n_hidden": 2,
        "layer_size": 64,
        "normalization": "layernorm",
        "hidden_factor": 2.0,
        "hidden_dropout": 0.1,
        "residual_dropout": 0.05,
        "batch_size": 256,
    }


def run_resnet_default(X, y) -> Dict[str, float]:
    """
    Run ResNet with default hyperparameters and K-fold cross validation.
    """
    print("=== ResNet (Default hyperparameters) ===")
    params = resnet_default_params()
    metrics = kfold_evaluate_resnet(params, X, y)
    print(metrics)
    return metrics


def resnet_optuna_objective(trial: optuna.Trial, X, y) -> float:
    """
    Optuna objective for ResNet single-objective optimization.

    The objective minimizes the average RMSE across both targets.
    """
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "n_hidden": trial.suggest_int("n_hidden", 1, 5),
        "layer_size": trial.suggest_int("layer_size", 16, 256, log=True),
        "normalization": trial.suggest_categorical("normalization", ["batchnorm", "layernorm"]),
        "hidden_factor": trial.suggest_float("hidden_factor", 1.0, 4.0),
        "hidden_dropout": trial.suggest_float("hidden_dropout", 0.0, 0.5),
        "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.5),
        "batch_size": trial.suggest_int("batch_size", 64, 512, log=True),
    }

    metrics = kfold_evaluate_resnet(params, X, y)
    # ***** CHANGED: optimize RMSE instead of MAE *****
    obj = (metrics["rmse_dom"] + metrics["rmse_price"]) / 2.0
    trial.set_user_attr("metrics", metrics)
    return obj


def run_resnet_optuna_soo(X, y, n_trials: int = 30):
    """
    Run single-objective Bayesian optimization for ResNet using Optuna.

    Returns
    -------
    metrics : dict
        Metrics for the best configuration.
    best_params : dict
        Selected hyperparameters for ResNet.
    pareto_points : (np.ndarray, np.ndarray)
        Pareto front points in RMSE space extracted from Optuna trials.
    """
    print("=== ResNet + Optuna (SOO) ===")
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="resnet_soo")
    study.optimize(lambda t: resnet_optuna_objective(t, X, y), n_trials=n_trials)

    best_trial = study.best_trial
    best_params = best_trial.params
    print("Best ResNet params (SOO):", best_params)
    metrics = kfold_evaluate_resnet(best_params, X, y)
    print("ResNet SOO metrics:", metrics)

    xs, ys = extract_rmse_points_from_trials(study.trials)
    xs_pf, ys_pf = pareto_front(xs, ys)

    return metrics, best_params, (xs_pf, ys_pf)


def run_resnet_optuna_moo_weighted(X, y, alphas=None, n_trials_per_alpha: int = 10):
    """
    Approximate multi-objective optimization for ResNet
    using a weighted sum of RMSE over several alpha values.
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=RANDOM_STATE)
    xs_all, ys_all = [], []
    metrics_per_alpha: Dict[float, Dict[str, float]] = {}

    for alpha in alphas:
        w1 = alpha ** 2
        w2 = 1.0 - w1

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "n_hidden": trial.suggest_int("n_hidden", 1, 5),
                "layer_size": trial.suggest_int("layer_size", 16, 256, log=True),
                "normalization": trial.suggest_categorical("normalization", ["batchnorm", "layernorm"]),
                "hidden_factor": trial.suggest_float("hidden_factor", 1.0, 4.0),
                "hidden_dropout": trial.suggest_float("hidden_dropout", 0.0, 0.5),
                "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.5),
                "batch_size": trial.suggest_int("batch_size", 64, 512, log=True),
            }

            metrics = kfold_evaluate_resnet(params, X, y)
            trial.set_user_attr("metrics", metrics)
            obj = w1 * metrics["rmse_dom"] + w2 * metrics["rmse_price"]
            return obj

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials_per_alpha)

        best_trial = study.best_trial
        m = best_trial.user_attrs["metrics"]
        metrics_per_alpha[alpha] = m
        xs_all.append(m["rmse_dom"])
        ys_all.append(m["rmse_price"])

    xs_all = np.array(xs_all)
    ys_all = np.array(ys_all)
    xs_pf, ys_pf = pareto_front(xs_all, ys_all)

    idx_best = np.argmin(xs_all + ys_all)
    best_metrics_table = list(metrics_per_alpha.values())[idx_best]

    return best_metrics_table, (xs_pf, ys_pf)


# ============================================================
# RESNET LEARNING CURVE & SCATTER PLOTS
# ============================================================

def plot_resnet_learning_curve(X, y, params: Dict) -> None:
    """
    Train a single ResNet model and plot the training/validation
    learning curve (RMSE vs. epoch).
    """
    K.clear_session()
    set_global_seeds(RANDOM_STATE + 100)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = construct_resnet_model(params, X_train.shape[-1])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=15,
        restore_best_weights=True,
        verbose=0,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        patience=5,
        factor=0.5,
        min_lr=1e-5,
        verbose=0,
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=params.get("batch_size", 256),
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    # Loss is MSE, convert to RMSE for plotting
    loss = np.array(history.history.get("loss", []))
    val_loss = np.array(history.history.get("val_loss", []))
    rmse = np.sqrt(loss)
    val_rmse = np.sqrt(val_loss)

    plt.figure()
    plt.plot(rmse, label="Train RMSE")
    plt.plot(val_rmse, label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("ResNet Learning Curve (RMSE)")
    plt.legend()
    plt.tight_layout()
    full_path = os.path.join(PLOT_DIR, "resnet_learning_curve.png")
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"Saved ResNet learning curve: {full_path}")


def plot_resnet_price_scatter(X, y, params: Dict) -> None:
    """
    Train a ResNet model and create a scatter plot of
    true vs predicted price.
    """
    K.clear_session()
    set_global_seeds(RANDOM_STATE + 200)

    model = construct_resnet_model(params, X.shape[-1])

    model.fit(
        X,
        y,
        batch_size=params.get("batch_size", 256),
        epochs=50,
        validation_split=0.2,
        verbose=0,
    )

    y_pred = model.predict(X, verbose=0)
    true_price = y[:, 1]
    pred_price = y_pred[:, 1]

    plt.figure()
    plt.scatter(true_price, pred_price, alpha=0.4)
    min_val = min(true_price.min(), pred_price.min())
    max_val = max(true_price.max(), pred_price.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    plt.xlabel("True price (standardized)")
    plt.ylabel("Predicted price (standardized)")
    plt.title("ResNet: True vs Predicted Price")
    plt.legend()
    plt.tight_layout()
    full_path = os.path.join(PLOT_DIR, "resnet_price_scatter.png")
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"Saved ResNet price scatter: {full_path}")


def plot_resnet_dom_scatter(X, y, params: Dict) -> None:
    """
    Train a ResNet model and create a scatter plot of
    true vs predicted Days on Market.
    """
    K.clear_session()
    set_global_seeds(RANDOM_STATE + 300)

    model = construct_resnet_model(params, X.shape[-1])

    model.fit(
        X,
        y,
        batch_size=params.get("batch_size", 256),
        epochs=50,
        validation_split=0.2,
        verbose=0,
    )

    y_pred = model.predict(X, verbose=0)
    true_dom = y[:, 0]
    pred_dom = y_pred[:, 0]

    plt.figure()
    plt.scatter(true_dom, pred_dom, alpha=0.4)
    min_val = min(true_dom.min(), pred_dom.min())
    max_val = max(true_dom.max(), pred_dom.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    plt.xlabel("True Days on Market (standardized)")
    plt.ylabel("Predicted Days on Market (standardized)")
    plt.title("ResNet: True vs Predicted Days on Market")
    plt.legend()
    plt.tight_layout()
    full_path = os.path.join(PLOT_DIR, "resnet_dom_scatter.png")
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"Saved ResNet DOM scatter: {full_path}")


# ============================================================
# TABLE UTILITIES (PANDAS DATAFRAMES)
# ============================================================

def metrics_to_row(model_name: str, m: Dict[str, float]) -> Dict[str, float]:
    """
    Convert a metrics dictionary into a single table row.
    """
    return {
        "Model": model_name,
        "R2_dom": m["r2_dom"],
        "R2_price": m["r2_price"],
        "RMSE_dom": m["rmse_dom"],
        "RMSE_price": m["rmse_price"],
        "MAE_dom": m["mae_dom"],
        "MAE_price": m["mae_price"],
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main() -> None:
    """
    Main entry point:
    - load and standardize data
    - run all models (baseline, SOO, MOO)
    - print summary tables
    - generate Pareto and ResNet plots
    """
    set_global_seeds(RANDOM_STATE)

    X, y = load_and_standardize(DATA_PATH)

    # 1) Baseline models with standardized features
    lr_metrics = run_linear_regression(X, y)
    xgb_def_metrics = run_xgb_default(X, y)
    resnet_def_metrics = run_resnet_default(X, y)

    df_default = pd.DataFrame([
        metrics_to_row("Linear Regression", lr_metrics),
        metrics_to_row("XGBoost", xgb_def_metrics),
        metrics_to_row("ResNet", resnet_def_metrics),
    ])
    print("\n=== Table 1: Default Hyperparameters with StandardScaler ===")
    print(df_default)

    # 2) XGBoost and ResNet with single-objective Bayesian optimization
    xgb_soo_metrics, xgb_best_config, (xgb_soo_x, xgb_soo_y) = run_xgb_optuna_soo(X, y, n_trials=30)
    resnet_soo_metrics, resnet_best_params, (resnet_soo_x, resnet_soo_y) = run_resnet_optuna_soo(X, y, n_trials=30)

    df_soo = pd.DataFrame([
        metrics_to_row("XGBoost", xgb_soo_metrics),
        metrics_to_row("ResNet", resnet_soo_metrics),
    ])
    print("\n=== Table 2: SOO with Optimized Hyperparameters (StandardScaler) ===")
    print(df_soo)

    # 3) XGBoost and ResNet with weighted-sum MOO approximation
    xgb_moo_metrics, (xgb_moo_x, xgb_moo_y) = run_xgb_optuna_moo_weighted(
        X, y, alphas=[0.0, 0.25, 0.5, 0.75, 1.0], n_trials_per_alpha=15
    )
    resnet_moo_metrics, (resnet_moo_x, resnet_moo_y) = run_resnet_optuna_moo_weighted(
        X, y, alphas=[0.0, 0.25, 0.5, 0.75, 1.0], n_trials_per_alpha=10
    )

    df_moo = pd.DataFrame([
        metrics_to_row("XGBoost", xgb_moo_metrics),
        metrics_to_row("ResNet", resnet_moo_metrics),
    ])
    print("\n=== Table 3: MOO with Optimized Hyperparameters (StandardScaler) ===")
    print(df_moo)

    # 4) Pareto plots comparing SOO and MOO results
    plot_pareto(
        xgb_soo_x,
        xgb_soo_y,
        xgb_moo_x,
        xgb_moo_y,
        title="Non-dominated Points using RMSE (XGBoost)",
        filename="pareto_xgb_rmse.png",
    )

    plot_pareto(
        resnet_soo_x,
        resnet_soo_y,
        resnet_moo_x,
        resnet_moo_y,
        title="Non-dominated Points using RMSE (ResNet)",
        filename="pareto_resnet_rmse.png",
    )

    # 5) Additional diagnostic plots for ResNet
    plot_resnet_learning_curve(X, y, resnet_best_params)
    plot_resnet_price_scatter(X, y, resnet_best_params)
    plot_resnet_dom_scatter(X, y, resnet_best_params)


if __name__ == "__main__":
    main()
