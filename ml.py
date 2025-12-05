# ml.py
import pandas as pd
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Complain",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
]


NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Complain",
    "Satisfaction Score",
    "Point Earned",
]

CATEGORICAL_FEATURES = [
    "Geography",
    "Gender",
    "Card Type",
]


def _prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Выделяет фичи и целевую переменную.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df["Exited"].astype(int)
    return X, y


def _build_preprocessor() -> ColumnTransformer:
    """
    Препроцессор: числовые — медианный имьютер; категориальные — most_frequent + OHE.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def _build_model(model_type: str):
    """
    Возвращает классическую модель по типу.
    """
    model_type = model_type.lower()
    if model_type == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")


def train_model(
    df: pd.DataFrame,
    model_type: str = "logreg",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Обучает модель (логистическая регрессия или случайный лес) и возвращает:
    - обученный пайплайн (preprocess + model)
    - метрики по тесту.
    """
    X, y = _prepare_features_target(df)
    preprocessor = _build_preprocessor()
    model = _build_model(model_type)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    return clf, metrics


def predict_for_customer(model, customer_row: pd.Series) -> Tuple[int, float]:
    """
    Делает прогноз для одного клиента:
    - возвращает предсказанный класс (0/1)
    - и вероятность оттока (Exited=1).
    """
    x_new = customer_row[FEATURE_COLUMNS].to_frame().T
    proba = model.predict_proba(x_new)[0, 1]
    pred = int(proba >= 0.5)
    return pred, float(proba)
