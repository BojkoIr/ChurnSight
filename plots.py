# plots.py
import pandas as pd
import plotly.express as px
from typing import List


def plot_bar_churn_by(df: pd.DataFrame, by: str, title: str):
    """
    Bar-график: доля ушедших клиентов по категории (регион, пол, активность и т.д.)
    """
    churn_rates = (
        df.groupby(by)["Exited"]
        .mean()
        .reset_index()
        .sort_values("Exited", ascending=False)
    )

    fig = px.bar(
        churn_rates,
        x=by,
        y="Exited",
        text="Exited",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        title=title,
        xaxis_title=by,
        yaxis_title="Доля ушедших",
        yaxis=dict(range=[0, 1]),
        margin=dict(t=60, l=40, r=20, b=60),
    )
    return fig


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Список числовых колонок для анализа распределений.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in ["RowNumber", "CustomerId"]:
        if col in numeric_cols:
            numeric_cols.remove(col)
    return numeric_cols


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]):
    """
    Корреляционная матрица (интерактивная тепловая карта).
    """
    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        origin="lower",
        aspect="auto",
    )
    fig.update_layout(
        title="Корреляционная матрица числовых признаков",
        xaxis_title="Признаки",
        yaxis_title="Признаки",
        margin=dict(t=60, l=60, r=20, b=60),
    )
    return fig


def plot_tenure_churn_bar(df: pd.DataFrame):
    """
    Отток в зависимости от стажа клиента (Tenure).
    """
    tenure_bins = [-1, 2, 5, 10]
    tenure_labels = ["0–2 года", "3–5 лет", "6–10 лет"]
    tenure_cat = pd.cut(df["Tenure"], bins=tenure_bins, labels=tenure_labels)

    tenure_churn = (
        df.assign(TenureCat=tenure_cat)
        .groupby("TenureCat")["Exited"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        tenure_churn,
        x="TenureCat",
        y="Exited",
        text="Exited",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        title="Отток в зависимости от стажа клиента",
        xaxis_title="Стаж клиента",
        yaxis_title="Доля ушедших",
        yaxis=dict(range=[0, 1]),
        margin=dict(t=60, l=40, r=20, b=60),
    )
    return fig
