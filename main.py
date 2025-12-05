# main.py
import streamlit as st
import pandas as pd
import plotly.express as px

from filters import filter_data
from plots import (
    plot_bar_churn_by,
    get_numeric_columns,
    plot_correlation_heatmap,
    plot_tenure_churn_bar,
)

# ==========================
# НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ==========================
st.set_page_config(
    page_title="Анализ оттока клиентов банка",
    layout="wide",
)

st.title("Анализ оттока клиентов банка")
st.markdown(
    "Интерактивный дашборд для исследования факторов, влияющих на уход клиентов (`Exited`), "
    "и простая модель прогноза оттока."
)


# ==========================
# ЗАГРУЗКА ДАННЫХ
# ==========================

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

DATA_PATH = "Customer-Churn-Records.csv"
data = load_data(DATA_PATH)


# ==========================
# KPI
# ==========================

def calc_kpis(df: pd.DataFrame) -> dict:
    kpis = {}
    kpis["total_customers"] = len(df)

    if len(df) > 0:
        kpis["churn_rate"] = df["Exited"].mean()
        kpis["avg_credit_score"] = df["CreditScore"].mean()
        kpis["avg_balance"] = df["Balance"].mean()
        kpis["avg_salary"] = df["EstimatedSalary"].mean()
        kpis["avg_tenure"] = df["Tenure"].mean()
    else:
        kpis["churn_rate"] = 0
        kpis["avg_credit_score"] = 0
        kpis["avg_balance"] = 0
        kpis["avg_salary"] = 0
        kpis["avg_tenure"] = 0

    return kpis


# ==========================
# САЙДБАР — ФИЛЬТРЫ
# ==========================

st.sidebar.header("Фильтры")

geography_options = sorted(data["Geography"].dropna().unique())
selected_geography = st.sidebar.multiselect(
    "Регион (Geography):",
    options=geography_options,
    default=geography_options,
)

gender_options = ["Все", "Male", "Female"]
selected_gender = st.sidebar.radio("Пол (Gender):", options=gender_options, index=0)

min_age = int(data["Age"].min())
max_age = int(data["Age"].max())
selected_age_range = st.sidebar.slider(
    "Возраст (Age):",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age),
)

min_tenure = int(data["Tenure"].min())
max_tenure = int(data["Tenure"].max())
selected_tenure_range = st.sidebar.slider(
    "Стаж клиента, лет (Tenure):",
    min_value=min_tenure,
    max_value=max_tenure,
    value=(min_tenure, max_tenure),
)

is_active_options = ["Все", "Только активные", "Только неактивные"]
selected_is_active = st.sidebar.selectbox(
    "Активность клиента (IsActiveMember):",
    options=is_active_options,
)

exited_options = ["Все", "Только ушедшие", "Только оставшиеся"]
selected_exited = st.sidebar.selectbox(
    "Статус клиента (Exited):",
    options=exited_options,
)

st.sidebar.markdown("---")
st.sidebar.caption("Датасет: Customer-Churn-Records.csv")


# ==========================
# ПРИМЕНЕНИЕ ФИЛЬТРОВ
# ==========================

filtered_data = filter_data(
    data,
    geography=selected_geography,
    gender=selected_gender,
    age_range=selected_age_range,
    tenure_range=selected_tenure_range,
    is_active_filter=selected_is_active,
    exited_filter=selected_exited,
)

if filtered_data.empty:
    st.warning("По заданным фильтрам данных не найдено. Попробуйте ослабить условия.")
    st.stop()


# ==========================
# KPI-БЛОК
# ==========================

kpis = calc_kpis(filtered_data)

st.subheader("Ключевые показатели по отфильтрованным данным")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Количество клиентов", f"{kpis['total_customers']}")
col2.metric("Отток (доля ушедших)", f"{kpis['churn_rate'] * 100:.1f} %")
col3.metric("Средний кредитный рейтинг", f"{kpis['avg_credit_score']:.0f}")
col4.metric("Средний баланс на счёте", f"{kpis['avg_balance']:.0f}")
col5.metric("Средняя зарплата", f"{kpis['avg_salary']:.0f}")

st.markdown("---")


# ==========================
# ВКЛАДКИ
# ==========================

tab_overview, tab_distributions, tab_factors, tab_data, tab_client = st.tabs(
    ["Обзор", "Распределения", "Факторы оттока", "Данные", "Профиль клиента"]
)


# --------------------------
# ТАБ "ОБЗОР"
# --------------------------
with tab_overview:
    st.subheader("Общий обзор оттока")

    col_a, col_b = st.columns(2)

    with col_a:
        fig = plot_bar_churn_by(filtered_data, by="Geography", title="Отток по регионам")
        st.plotly_chart(fig, width="stretch")

    with col_b:
        fig = plot_bar_churn_by(filtered_data, by="Gender", title="Отток по полу")
        st.plotly_chart(fig, width="stretch")

    st.markdown("-----")

    col_c, col_d = st.columns(2)

    with col_c:
        fig = plot_bar_churn_by(
            filtered_data,
            by="IsActiveMember",
            title="Отток в зависимости от активности",
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["Неактивные (0)", "Активные (1)"],
        )
        st.plotly_chart(fig, width="stretch")

    with col_d:
        fig = plot_bar_churn_by(
            filtered_data,
            by="HasCrCard",
            title="Отток в зависимости от наличия карты",
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["Нет карты (0)", "Есть карта (1)"],
        )
        st.plotly_chart(fig, width="stretch")


# --------------------------
# ТАБ "РАСПРЕДЕЛЕНИЯ"
# --------------------------
with tab_distributions:
    st.subheader("Распределения числовых признаков")

    numeric_cols = get_numeric_columns(filtered_data)

    col_top1, col_top2 = st.columns([2, 1])
    with col_top1:
        selected_feature = st.selectbox(
            "Выберите числовой признак для анализа:",
            options=numeric_cols,
        )
    with col_top2:
        bins = st.slider(
            "Количество корзин (bins):",
            min_value=5,
            max_value=80,
            value=40,
            step=5,
        )

    # аккуратное поле статуса
    filtered_plot = filtered_data.copy()
    filtered_plot["Status"] = filtered_plot["Exited"].map(
        {0: "Остались (0)", 1: "Ушли (1)"}
    )

    st.markdown("### Гистограмма + boxplot (интерактивно)")

    fig = px.histogram(
        filtered_plot,
        x=selected_feature,
        color="Status",
        nbins=bins,
        barmode="overlay",
        opacity=0.7,
        marginal="box",
        hover_data=["Geography", "Gender", "Age"],
    )
    fig.update_layout(
        xaxis_title=selected_feature,
        yaxis_title="Количество клиентов",
        legend_title="Статус",
        bargap=0.05,
        margin=dict(t=60, l=40, r=20, b=60),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Краткая статистика по признаку")

    col_stats1, col_stats2 = st.columns(2)

    with col_stats1:
        st.write(filtered_data[selected_feature].describe().to_frame("Значение"))

    with col_stats2:
        st.markdown("**Распределение по статусу клиента:**")
        fig_box = px.box(
            filtered_plot,
            x="Status",
            y=selected_feature,
            points="outliers",
        )
        fig_box.update_layout(
            xaxis_title="Статус клиента",
            yaxis_title=selected_feature,
            margin=dict(t=40, l=40, r=20, b=60),
        )
        st.plotly_chart(fig_box, use_container_width=True)


# --------------------------
# ТАБ "ФАКТОРЫ ОТТОКА"
# --------------------------
with tab_factors:
    st.subheader("Факторы оттока")

    st.markdown("**Корреляции числовых признаков (включая Exited):**")
    numeric_cols_full = get_numeric_columns(filtered_data)
    if "Exited" not in numeric_cols_full:
        numeric_cols_full.append("Exited")

    fig_corr = plot_correlation_heatmap(filtered_data, numeric_cols_full)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.markdown("**Отток в зависимости от стажа клиента (Tenure):**")

    fig_tenure = plot_tenure_churn_bar(filtered_data)
    st.plotly_chart(fig_tenure, use_container_width=True)


# --------------------------
# ТАБ "ДАННЫЕ"
# --------------------------
with tab_data:
    st.subheader("Отфильтрованные данные")
    st.caption(
        "Строки после применения фильтров. Можно выгрузить в CSV для дальнейшего анализа."
    )

    st.dataframe(filtered_data)

    csv = filtered_data.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Скачать отфильтрованные данные в CSV",
        data=csv,
        file_name="filtered_customer_churn.csv",
        mime="text/csv",
    )


# --------------------------
# ТАБ "ПРОФИЛЬ КЛИЕНТА"
# --------------------------
with tab_client:
    st.subheader("Профиль клиента и риск оттока по историческим данным")

    st.caption(
        "Можно проанализировать существующего клиента из базы или задать профиль нового клиента "
        "и при желании сохранить его в CSV."
    )

    # Базовый средний отток по всей выборке
    base_churn = data["Exited"].mean()

    # Немного статистики для разумных диапазонов
    cs_min, cs_max = int(data["CreditScore"].min()), int(data["CreditScore"].max())
    age_min, age_max = int(data["Age"].min()), int(data["Age"].max())
    bal_min, bal_max = 0, float(data["Balance"].max())
    sal_min, sal_max = 0, float(data["EstimatedSalary"].max())
    ten_min, ten_max = int(data["Tenure"].min()), int(data["Tenure"].max())
    prod_min, prod_max = int(data["NumOfProducts"].min()), int(data["NumOfProducts"].max())
    point_min, point_max = float(data["Point Earned"].min()), float(data["Point Earned"].max())

    geos = sorted(data["Geography"].unique())
    genders = sorted(data["Gender"].unique())
    card_types = sorted(data["Card Type"].unique())

    mode = st.radio(
        "Режим работы:",
        ["Клиент из базы", "Новый клиент"],
        horizontal=True,
    )

    # ==========================
    # 1. КЛИЕНТ ИЗ БАЗЫ
    # ==========================
    if mode == "Клиент из базы":
        st.markdown("### Выбор клиента из датасета")

        if "CustomerId" not in data.columns:
            st.error("В датасете нет поля CustomerId, невозможен выбор клиента.")
        else:
            customer_ids = data["CustomerId"].tolist()
            selected_customer_id = st.selectbox(
                "Выберите клиента по CustomerId:",
                options=customer_ids,
            )

            row = data[data["CustomerId"] == selected_customer_id].iloc[0]

            # Берём значения так же, как в ручном вводе
            credit_new = float(row["CreditScore"])
            geo_new = row["Geography"]
            gender_new = row["Gender"]
            age_new = int(row["Age"])
            tenure_new = int(row["Tenure"])
            balance_new = float(row["Balance"])
            salary_new = float(row["EstimatedSalary"])
            num_products_new = int(row["NumOfProducts"])
            has_cr_card_new = int(row["HasCrCard"])
            is_active_new = int(row["IsActiveMember"])
            complain_new = float(row["Complain"])
            satisfaction_new = float(row["Satisfaction Score"])
            card_type_new = row["Card Type"]
            points_new = float(row["Point Earned"])

            if st.button("Проанализировать выбранного клиента"):
                # --- тот же анализ, что и для нового клиента ---

                # 1. Сегмент
                segment = data[
                    (data["Geography"] == geo_new)
                    & (data["IsActiveMember"] == is_active_new)
                    & (data["NumOfProducts"] == num_products_new)
                ]
                segment_level = "Geography + IsActiveMember + NumOfProducts"

                if len(segment) < 40:
                    segment = data[
                        (data["Geography"] == geo_new)
                        & (data["IsActiveMember"] == is_active_new)
                    ]
                    segment_level = "Geography + IsActiveMember"

                if len(segment) < 40:
                    segment = data[data["Geography"] == geo_new]
                    segment_level = "Geography"

                if len(segment) == 0:
                    segment = data
                    segment_level = "вся база"

                segment_churn = segment["Exited"].mean()

                if segment_churn < 0.15:
                    risk_label = "Низкий риск"
                    risk_color = "🟢"
                elif segment_churn < 0.30:
                    risk_label = "Средний риск"
                    risk_color = "🟠"
                else:
                    risk_label = "Повышенный риск"
                    risk_color = "🔴"

                st.markdown("### Исторические данные по похожим клиентам")

                st.markdown(
                    f"- Уровень сегмента: **{segment_level}**  \n"
                    f"- Размер сегмента: **{len(segment)} клиентов**  \n"
                    f"- Средний отток в сегменте: **{segment_churn*100:.1f}%**  \n"
                    f"- Средний отток по всей базе: **{base_churn*100:.1f}%**"
                )

                st.markdown(f"**Итоговая оценка:** {risk_color} **{risk_label}**")

                st.markdown("### Ключевые факторы профиля клиента")
                factors = []

                if geo_new == "Germany":
                    factors.append("Клиент из Germany — в этом регионе исторически самый высокий отток.")
                if is_active_new == 0:
                    factors.append("Клиент неактивен (IsActiveMember=0) — это сильно повышает риск.")
                if num_products_new >= 3:
                    factors.append("У клиента много продуктов (NumOfProducts ≥ 3) — такие чаще уходят.")
                if credit_new < 600:
                    factors.append("Низкий кредитный рейтинг (CreditScore < 600).")
                if age_new >= 40:
                    factors.append("Возраст 40+ — это сегмент с повышенным риском оттока.")
                if satisfaction_new < 3:
                    factors.append("Низкий уровень удовлетворённости (Satisfaction Score < 3).")
                if complain_new > 0:
                    factors.append("Есть жалобы клиента (Complain > 0).")
                if balance_new > data["Balance"].median():
                    factors.append("Высокий баланс — важно удержать такого клиента.")

                if not factors:
                    factors.append("Явных факторов повышенного риска не выявлено по простым правилам.")

                for f in factors:
                    st.write(f"- {f}")

                st.markdown("### Позиция клиента относительно всей базы (перцентили)")

                rows = []
                for col, val, label in [
                    ("CreditScore", credit_new, "Кредитный рейтинг"),
                    ("Age", age_new, "Возраст"),
                    ("Balance", balance_new, "Баланс на счёте"),
                    ("EstimatedSalary", salary_new, "Зарплата"),
                    ("Tenure", tenure_new, "Стаж клиента"),
                    ("Satisfaction Score", satisfaction_new, "Удовлетворённость"),
                    ("Point Earned", points_new, "Накопленные баллы"),
                ]:
                    pct = (data[col] <= val).mean()
                    rows.append(
                        {
                            "Показатель": label,
                            "Значение клиента": round(val, 2),
                            "Перцентиль": f"{pct*100:.1f}%",
                        }
                    )

                st.table(pd.DataFrame(rows))

                st.markdown("### Клиент на фоне распределения CreditScore")
                fig_hist = px.histogram(
                    data,
                    x="CreditScore",
                    nbins=40,
                    opacity=0.75,
                )
                fig_hist.add_vline(
                    x=credit_new,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Клиент",
                    annotation_position="top"
                )
                fig_hist.update_layout(
                    xaxis_title="CreditScore",
                    yaxis_title="Количество клиентов",
                )
                st.plotly_chart(fig_hist, width="stretch")

                with st.expander("Сырые данные клиента"):
                    st.write(row)

    # ==========================
    # 2. НОВЫЙ КЛИЕНТ
    # ==========================
    else:
        st.markdown("### Новый клиент")

        with st.form("client_profile_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                geo_new = st.selectbox("Geography", options=geos)
                gender_new = st.selectbox("Gender", options=genders)
                age_new = st.slider(
                    "Age", min_value=age_min, max_value=age_max,
                    value=int(data["Age"].median())
                )
                tenure_new = st.slider(
                    "Tenure (лет в банке)",
                    min_value=ten_min, max_value=ten_max,
                    value=int(data["Tenure"].median())
                )

            with col2:
                credit_new = st.slider(
                    "CreditScore",
                    min_value=cs_min,
                    max_value=cs_max,
                    value=int(data["CreditScore"].median()),
                )
                balance_new = st.slider(
                    "Balance",
                    min_value=float(bal_min),
                    max_value=float(bal_max),
                    value=float(data["Balance"].median()),
                )
                salary_new = st.slider(
                    "EstimatedSalary",
                    min_value=float(sal_min),
                    max_value=float(sal_max),
                    value=float(data["EstimatedSalary"].median()),
                )

            with col3:
                num_products_new = st.slider(
                    "NumOfProducts",
                    min_value=prod_min,
                    max_value=prod_max,
                    value=int(data["NumOfProducts"].median()),
                )
                has_cr_card_new = st.radio("HasCrCard", options=[0, 1], index=1)
                is_active_new = st.radio("IsActiveMember", options=[0, 1], index=1)
                complain_new = st.slider(
                    "Complain (0/1 или счётчик жалоб)",
                    min_value=float(data["Complain"].min()),
                    max_value=float(data["Complain"].max()),
                    value=float(data["Complain"].median()),
                )
                satisfaction_new = st.slider(
                    "Satisfaction Score",
                    min_value=float(data["Satisfaction Score"].min()),
                    max_value=float(data["Satisfaction Score"].max()),
                    value=float(data["Satisfaction Score"].median()),
                )
                card_type_new = st.selectbox("Card Type", options=card_types)
                points_new = st.slider(
                    "Point Earned",
                    min_value=point_min,
                    max_value=point_max,
                    value=float(data["Point Earned"].median()),
                )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                analyze_clicked = st.form_submit_button(
                    "Проанализировать профиль клиента",
                    type="secondary"
                )
            with col_btn2:
                analyze_and_save_clicked = st.form_submit_button(
                    "Проанализировать и сохранить клиента в базу",
                    type="primary"
                )

        submitted = analyze_clicked or analyze_and_save_clicked

        if submitted:
            client = {
                "CreditScore": credit_new,
                "Geography": geo_new,
                "Gender": gender_new,
                "Age": age_new,
                "Tenure": tenure_new,
                "Balance": balance_new,
                "NumOfProducts": num_products_new,
                "HasCrCard": has_cr_card_new,
                "IsActiveMember": is_active_new,
                "EstimatedSalary": salary_new,
                "Complain": complain_new,
                "Satisfaction Score": satisfaction_new,
                "Card Type": card_type_new,
                "Point Earned": points_new,
            }

            # --- дальше тот же анализ, что и выше ---
            segment = data[
                (data["Geography"] == geo_new)
                & (data["IsActiveMember"] == is_active_new)
                & (data["NumOfProducts"] == num_products_new)
            ]
            segment_level = "Geography + IsActiveMember + NumOfProducts"

            if len(segment) < 40:
                segment = data[
                    (data["Geography"] == geo_new)
                    & (data["IsActiveMember"] == is_active_new)
                ]
                segment_level = "Geography + IsActiveMember"

            if len(segment) < 40:
                segment = data[data["Geography"] == geo_new]
                segment_level = "Geography"

            if len(segment) == 0:
                segment = data
                segment_level = "вся база"

            segment_churn = segment["Exited"].mean()

            if segment_churn < 0.15:
                risk_label = "Низкий риск"
                risk_color = "🟢"
            elif segment_churn < 0.30:
                risk_label = "Средний риск"
                risk_color = "🟠"
            else:
                risk_label = "Повышенный риск"
                risk_color = "🔴"

            st.markdown("### Исторические данные по похожим клиентам")

            st.markdown(
                f"- Уровень сегмента: **{segment_level}**  \n"
                f"- Размер сегмента: **{len(segment)} клиентов**  \n"
                f"- Средний отток в сегменте: **{segment_churn*100:.1f}%**  \n"
                f"- Средний отток по всей базе: **{base_churn*100:.1f}%**"
            )

            st.markdown(f"**Итоговая оценка:** {risk_color} **{risk_label}**")

            st.markdown("### Ключевые факторы профиля клиента")
            factors = []

            if geo_new == "Germany":
                factors.append("Клиент из Germany — в этом регионе исторически самый высокий отток.")
            if is_active_new == 0:
                factors.append("Клиент неактивен (IsActiveMember=0) — это сильно повышает риск.")
            if num_products_new >= 3:
                factors.append("У клиента много продуктов (NumOfProducts ≥ 3) — такие чаще уходят.")
            if credit_new < 600:
                factors.append("Низкий кредитный рейтинг (CreditScore < 600).")
            if age_new >= 40:
                factors.append("Возраст 40+ — это сегмент с повышенным риском оттока.")
            if satisfaction_new < 3:
                factors.append("Низкий уровень удовлетворённости (Satisfaction Score < 3).")
            if complain_new > 0:
                factors.append("Есть жалобы клиента (Complain > 0).")
            if balance_new > data["Balance"].median():
                factors.append("Высокий баланс — важно удержать такого клиента.")

            if not factors:
                factors.append("Явных факторов повышенного риска не выявлено по простым правилам.")

            for f in factors:
                st.write(f"- {f}")

            st.markdown("### Позиция клиента относительно всей базы (перцентили)")
            rows = []
            for col, val, label in [
                ("CreditScore", credit_new, "Кредитный рейтинг"),
                ("Age", age_new, "Возраст"),
                ("Balance", balance_new, "Баланс на счёте"),
                ("EstimatedSalary", salary_new, "Зарплата"),
                ("Tenure", tenure_new, "Стаж клиента"),
                ("Satisfaction Score", satisfaction_new, "Удовлетворённость"),
                ("Point Earned", points_new, "Накопленные баллы"),
            ]:
                pct = (data[col] <= val).mean()
                rows.append(
                    {
                        "Показатель": label,
                        "Значение клиента": round(val, 2),
                        "Перцентиль": f"{pct*100:.1f}%",
                    }
                )
            st.table(pd.DataFrame(rows))

            st.markdown("### Клиент на фоне распределения CreditScore")
            fig_hist = px.histogram(
                data,
                x="CreditScore",
                nbins=40,
                opacity=0.75,
            )
            fig_hist.add_vline(
                x=credit_new,
                line_dash="dash",
                line_color="red",
                annotation_text="Клиент",
                annotation_position="top"
            )
            fig_hist.update_layout(
                xaxis_title="CreditScore",
                yaxis_title="Количество клиентов",
            )
            st.plotly_chart(fig_hist, width="stretch")

            with st.expander("Параметры клиента (как словарь)"):
                st.write(client)

            # ===== Сохранение клиента в базу =====
            if analyze_and_save_clicked:
                new_id = int(data["CustomerId"].max()) + 1 if "CustomerId" in data.columns else 1
                client_row = client.copy()
                client_row["CustomerId"] = new_id
                client_row["Exited"] = None  # статус неизвестен для нового

                updated_data = pd.concat(
                    [data, pd.DataFrame([client_row])],
                    ignore_index=True
                )
                updated_data.to_csv(DATA_PATH, index=False)

                # "Всплывающее" уведомление + обычный статус
                if hasattr(st, "toast"):
                    st.toast(f"Клиент сохранён. ID: {new_id}", icon="✅")
                st.success(f"Клиент сохранён в файл. Присвоен ID: {new_id}")
                st.info("После перезапуска приложения клиент появится в разделе «Данные» и на графиках.")
