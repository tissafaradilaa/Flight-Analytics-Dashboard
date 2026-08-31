import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Flight Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 17px;
        margin-top: 4px;
        margin-bottom: 30px;
    }

    /* KPI Card */
    .kpi-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        height: 135px;
    }

    .kpi-title {
        color: #6b7280;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: #111827;
        font-size: 30px;
        font-weight: 800;
    }

    .kpi-description {
        color: #9ca3af;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Section title */
    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #111827;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    /* Chart card */
    .chart-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }

    /* ML cards */
    .ml-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: #9ca3af !important;
        margin-bottom: 25px;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 30px 0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    file_path = r"D:\Tissa\visualisasi_data\flight\flights_sample_10000.csv"

    df = pd.read_csv(file_path)

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    return df


df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">✈️ Flight Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Interactive flight performance dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🔎 Filters")

    airlines = sorted(
        df["AIRLINE"]
        .dropna()
        .unique()
    )

    selected_airlines = st.multiselect(
        "Airline",
        airlines,
        default=airlines
    )

    min_date = df["FL_DATE"].min()
    max_date = df["FL_DATE"].max()

    selected_date = st.date_input(
        "Flight Date",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    df["AIRLINE"].isin(selected_airlines)
].copy()


# Date filtering

if isinstance(selected_date, tuple) and len(selected_date) == 2:

    start_date = pd.Timestamp(selected_date[0])
    end_date = pd.Timestamp(selected_date[1])

    filtered_df = filtered_df[
        (filtered_df["FL_DATE"] >= start_date)
        &
        (filtered_df["FL_DATE"] <= end_date)
    ]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Flight Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore flight delays, cancellations, airline performance, '
    'and machine learning predictions.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# KPI
# ============================================================

total_flights = len(filtered_df)

cancelled_flights = filtered_df["CANCELLED"].sum()

delayed_flights = (
    filtered_df["DEP_DELAY"]
    .dropna()
    .gt(15)
    .sum()
)

avg_delay = filtered_df["DEP_DELAY"].mean()

cancellation_rate = (
    cancelled_flights / total_flights * 100
    if total_flights > 0
    else 0
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">TOTAL FLIGHTS</div>
            <div class="kpi-value">{total_flights:,}</div>
            <div class="kpi-description">
                Flights in selected period
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">DELAYED FLIGHTS</div>
            <div class="kpi-value">{delayed_flights:,}</div>
            <div class="kpi-description">
                Delay greater than 15 minutes
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">CANCELLED</div>
            <div class="kpi-value">{int(cancelled_flights):,}</div>
            <div class="kpi-description">
                Cancellation rate {cancellation_rate:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">AVG. DELAY</div>
            <div class="kpi-value">{avg_delay:.1f} min</div>
            <div class="kpi-description">
                Average departure delay
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Overview",
    "🤖 Machine Learning",
    "📄 Data"
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        '📈 Flight Performance'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MONTHLY TREND
    # --------------------------------------------------------

    filtered_df["Month"] = (
        filtered_df["FL_DATE"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_stats = (
        filtered_df
        .groupby("Month")
        .agg({
            "CANCELLED": "sum",
            "DEP_DELAY": lambda x: (x > 15).sum()
        })
        .reset_index()
    )

    fig1, ax1 = plt.subplots(
        figsize=(12, 5)
    )

    ax1.plot(
        monthly_stats["Month"],
        monthly_stats["DEP_DELAY"],
        marker="o",
        linewidth=2,
        label="Delayed >15 min"
    )

    ax1.plot(
        monthly_stats["Month"],
        monthly_stats["CANCELLED"],
        marker="x",
        linewidth=2,
        label="Cancelled"
    )

    ax1.set_title(
        "Monthly Flight Delay & Cancellation Trend",
        fontsize=14,
        fontweight="bold"
    )

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Number of Flights")

    ax1.grid(
        alpha=0.2
    )

    ax1.legend(
        frameon=False
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(
        fig1,
        use_container_width=True
    )


    # --------------------------------------------------------
    # TWO COLUMN CHARTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # DELAY CAUSES
    # --------------------------------------------------------

    with col1:

        st.markdown(
            '<div class="section-title">'
            '⏱️ Delay Causes'
            '</div>',
            unsafe_allow_html=True
        )

        delay_cols = [
            "DELAY_DUE_CARRIER",
            "DELAY_DUE_WEATHER",
            "DELAY_DUE_NAS",
            "DELAY_DUE_SECURITY",
            "DELAY_DUE_LATE_AIRCRAFT"
        ]

        total_delays = (
            filtered_df[delay_cols]
            .sum()
            .sort_values()
        )

        fig2, ax2 = plt.subplots(
            figsize=(8, 5)
        )

        total_delays.plot(
            kind="barh",
            ax=ax2
        )

        ax2.set_title(
            "Total Delay by Cause",
            fontsize=14,
            fontweight="bold"
        )

        ax2.set_xlabel(
            "Total Delay Minutes"
        )

        ax2.set_ylabel("")

        ax2.grid(
            axis="x",
            alpha=0.2
        )

        plt.tight_layout()

        st.pyplot(
            fig2,
            use_container_width=True
        )


    # --------------------------------------------------------
    # AIRLINE PERFORMANCE
    # --------------------------------------------------------

    with col2:

        st.markdown(
            '<div class="section-title">'
            '✈️ Airline Performance'
            '</div>',
            unsafe_allow_html=True
        )

        airline_delay = (
            filtered_df[
                ["AIRLINE", "DEP_DELAY"]
            ]
            .dropna()
        )

        airline_avg_delay = (
            airline_delay
            .groupby("AIRLINE")["DEP_DELAY"]
            .mean()
            .sort_values()
        )

        fig3, ax3 = plt.subplots(
            figsize=(8, 5)
        )

        sns.barplot(
            x=airline_avg_delay.values,
            y=airline_avg_delay.index,
            ax=ax3
        )

        ax3.set_title(
            "Average Departure Delay",
            fontsize=14,
            fontweight="bold"
        )

        ax3.set_xlabel(
            "Average Delay (minutes)"
        )

        ax3.set_ylabel("")

        ax3.grid(
            axis="x",
            alpha=0.2
        )

        plt.tight_layout()

        st.pyplot(
            fig3,
            use_container_width=True
        )


# ============================================================
# TAB 2 — MACHINE LEARNING
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">'
        '🤖 Flight Delay Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    model_df = filtered_df.copy()

    model_df["DAY_OF_WEEK"] = (
        model_df["FL_DATE"]
        .dt.dayofweek
    )

    model_df["MONTH"] = (
        model_df["FL_DATE"]
        .dt.month
    )

    model_df["DELAYED"] = (
        model_df["DEP_DELAY"] > 15
    )


    X = model_df[
        [
            "AIRLINE",
            "ORIGIN",
            "DEST",
            "DAY_OF_WEEK",
            "MONTH"
        ]
    ]

    y = model_df["DELAYED"]


    categorical_cols = [
        "AIRLINE",
        "ORIGIN",
        "DEST"
    ]

    numerical_cols = [
        "DAY_OF_WEEK",
        "MONTH"
    ]


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_cols
            ),
            (
                "num",
                SimpleImputer(
                    strategy="mean"
                ),
                numerical_cols
            )
        ]
    )


    clf = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    n_estimators=100
                )
            )
        ]
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    clf.fit(
        X_train,
        y_train
    )

    y_pred = clf.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # MODEL KPI
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model Accuracy",
            f"{accuracy:.2%}"
        )

    with col2:

        st.metric(
            "Training Data",
            f"{len(X_train):,}"
        )

    with col3:

        st.metric(
            "Testing Data",
            f"{len(X_test):,}"
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### 🎯 Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig4, ax4 = plt.subplots(
            figsize=(6, 5)
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "On Time",
                "Delayed"
            ]
        )

        disp.plot(
            ax=ax4,
            cmap="Blues"
        )

        ax4.set_title(
            "Prediction Performance"
        )

        st.pyplot(
            fig4,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    with col2:

        st.markdown(
            "### 📊 Prediction Distribution"
        )

        prediction_counts = (
            pd.Series(y_pred)
            .value_counts()
        )

        prediction_counts.index = [
            "Delayed" if x
            else "On Time"
            for x in prediction_counts.index
        ]

        fig5, ax5 = plt.subplots(
            figsize=(6, 5)
        )

        sns.barplot(
            x=prediction_counts.index,
            y=prediction_counts.values,
            ax=ax5
        )

        ax5.set_title(
            "Predicted Flight Status"
        )

        ax5.set_xlabel("")
        ax5.set_ylabel("Number of Flights")

        st.pyplot(
            fig5,
            use_container_width=True
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        "### 🔍 Top 15 Feature Importance"
    )

    ohe = (
        clf
        .named_steps["preprocessor"]
        .named_transformers_["cat"]
    )

    ohe_features = (
        ohe.get_feature_names_out(
            categorical_cols
        )
    )

    all_features = np.concatenate(
        [
            ohe_features,
            numerical_cols
        ]
    )

    importances = (
        clf
        .named_steps["classifier"]
        .feature_importances_
    )

    feat_importance = (
        pd.Series(
            importances,
            index=all_features
        )
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    fig6, ax6 = plt.subplots(
        figsize=(10, 6)
    )

    feat_importance.sort_values().plot(
        kind="barh",
        ax=ax6
    )

    ax6.set_title(
        "Most Important Features"
    )

    ax6.set_xlabel(
        "Importance"
    )

    ax6.set_ylabel("")

    st.pyplot(
        fig6,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.markdown(
        "### 📋 Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "On Time",
            "Delayed"
        ],
        output_dict=True
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    st.dataframe(
        report_df.style.format(
            {
                "precision": "{:.2f}",
                "recall": "{:.2f}",
                "f1-score": "{:.2f}",
                "support": "{:.0f}"
            }
        ),
        use_container_width=True
    )


# ============================================================
# TAB 3 — DATA
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">'
        '📄 Flight Dataset'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Showing **{len(filtered_df):,}** flights"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=500
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#9ca3af; font-size:13px;">
        ✈️ Flight Analytics Dashboard &nbsp;•&nbsp;
        Data Visualization & Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)