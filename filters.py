# filters.py
import pandas as pd
from typing import List, Tuple


def filter_data(
    df: pd.DataFrame,
    geography: List[str],
    gender: str,
    age_range: Tuple[int, int],
    tenure_range: Tuple[int, int],
    is_active_filter: str,
    exited_filter: str,
) -> pd.DataFrame:
    """
    Применяет все фильтры к датафрейму.
    """
    filtered = df.copy()

    # География
    if geography:
        filtered = filtered[filtered["Geography"].isin(geography)]

    # Пол
    if gender != "Все":
        filtered = filtered[filtered["Gender"] == gender]

    # Возраст
    filtered = filtered[filtered["Age"].between(age_range[0], age_range[1])]

    # Стаж (Tenure)
    filtered = filtered[filtered["Tenure"].between(tenure_range[0], tenure_range[1])]

    # Активность
    if is_active_filter != "Все":
        active_value = 1 if is_active_filter == "Только активные" else 0
        filtered = filtered[filtered["IsActiveMember"] == active_value]

    # Статус оттока
    if exited_filter != "Все":
        exited_value = 1 if exited_filter == "Только ушедшие" else 0
        filtered = filtered[filtered["Exited"] == exited_value]

    return filtered
