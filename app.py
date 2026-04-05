"""
End-to-End Data Science Dashboard — Mini AutoML
================================================
Upload → Explore → Visualize → Train → Predict → Download
Built with Streamlit, Pandas, Plotly, Scikit-Learn, and XGBoost.
"""

import io
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AutoML Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>


/* ---------- Glassmorphic cards ---------- */
div[data-testid="stMetric"] {
    background: rgba(108, 99, 255, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.25);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.15);
    overflow: hidden;
}
div[data-testid="stMetric"] label {
    font-weight: 600;
    letter-spacing: 0.3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ---------- File uploader fix ---------- */
div[data-testid="stFileUploader"] button * {
    opacity: 0;
}
div[data-testid="stFileUploader"] button::after {
    content: "Browse files";
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    color: white;
    font-size: 14px;
    font-weight: 500;
}
div[data-testid="stFileUploader"] section {
    overflow: hidden;
}
div[data-testid="stFileUploaderDropzone"] {
    padding: 16px;
}

/* ---------- Expander fix ---------- */
div[data-testid="stExpander"] summary span {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ---------- Sidebar polish ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141a 0%, #1a1d26 100%);
    border-right: 1px solid rgba(108,99,255,0.15);
}
section[data-testid="stSidebar"] .stRadio label {
    padding: 6px 12px;
    border-radius: 8px;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(108, 99, 255, 0.12);
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #896bff 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.8rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    transition: transform 0.15s, box-shadow 0.25s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(108, 99, 255, 0.4);
    color: white;
}

/* ---------- Dataframe styling ---------- */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- Tab styling ---------- */
button[data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* ---------- Divider ---------- */
.gradient-divider {
    height: 3px;
    background: linear-gradient(90deg, #6C63FF 0%, #896bff 40%, transparent 100%);
    border: none;
    border-radius: 3px;
    margin: 12px 0 28px;
}

/* ---------- Header badge ---------- */
.header-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6C63FF 0%, #896bff 100%);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.DataFrame()


def infer_task(series: pd.Series) -> str:
    """Decide Classification vs Regression from target column."""
    if series.dtype == "object" or series.dtype.name == "category":
        return "classification"
    if series.nunique() <= 20 and series.nunique() / len(series) < 0.05:
        return "classification"
    return "regression"


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals, fill missing, return numeric matrix."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median()) if pd.api.types.is_numeric_dtype(X[col]) else X[col].fillna(X[col].mode().iloc[0])
    return X


CLASSIFICATION_MODELS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine": lambda: SVC(probability=True, random_state=42),
}

REGRESSION_MODELS = {
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": lambda: LinearRegression(),
    "Support Vector Machine": lambda: SVR(),
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS["XGBoost"] = lambda: XGBClassifier(
        n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    REGRESSION_MODELS["XGBoost"] = lambda: XGBRegressor(
        n_estimators=100, random_state=42
    )


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="header-badge">MINI AUTOML</p>', unsafe_allow_html=True)
    st.title("🧠 Data Science Dashboard")
    st.caption("Upload → Explore → Visualize → Train → Predict")
    st.markdown("---")

    page = st.radio(
        "**Navigate**",
        [
            "📂  Upload & Clean",
            "🔍  Auto EDA",
            "📊  Visualizations",
            "⚙️  Train Model",
            "🎯  Predict & Download",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;opacity:0.45;font-size:0.75rem;'>"
        "Built with ❤️ using Streamlit<br>© 2026 AutoML Dashboard</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  1 ▸ DATA UPLOAD & CLEANING
# ══════════════════════════════════════════════════════════════
if page.startswith("📂"):
    st.markdown('<p class="header-badge">STEP 1</p>', unsafe_allow_html=True)
    st.title("Upload & Clean Your Dataset")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Max 200 MB",
    )

    if uploaded is not None:
        file_id = getattr(uploaded, "file_id", str(uploaded.size) + uploaded.name)
        if st.session_state.get("last_file_id") != file_id:
            with st.spinner("Loading data …"):
                df = load_data(uploaded)
            st.session_state["df"] = df
            st.session_state["file_name"] = uploaded.name
            st.session_state["last_file_id"] = file_id
        
        df_current = st.session_state.get("df", pd.DataFrame())
        st.success(f"✅  **{uploaded.name}** active — {df_current.shape[0]:,} rows × {df_current.shape[1]} columns")
    
    if "df" in st.session_state and st.session_state["df"] is not None:
        if "msg_success" in st.session_state:
            st.success(st.session_state["msg_success"])
            del st.session_state["msg_success"]

        df = st.session_state["df"]

        # ── Metrics Row ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{df.shape[0]:,}")
        m2.metric("Columns", f"{df.shape[1]}")
        m3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
        m4.metric("Duplicated Rows", f"{df.duplicated().sum():,}")

        st.markdown("#### Preview")
        st.dataframe(df.head(30), use_container_width=True)

        # ── Data Types ──
        with st.expander("Column Data Types", expanded=False):
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Non-Null": df.notnull().sum().values,
                "Unique": df.nunique().values,
            })
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        # ── Cleaning Actions ──
        st.markdown("#### 🧹 Data Cleaning")
        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("Fill Missing (Median / Mode)", use_container_width=True):
                for col in df.columns:
                    if df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            mode_val = df[col].mode()
                            if not mode_val.empty:
                                df[col] = df[col].fillna(mode_val.iloc[0])
                st.session_state["df"] = df
                st.session_state["msg_success"] = "Missing values filled!"
                st.rerun()

        with c2:
            if st.button("Drop Duplicate Rows", use_container_width=True):
                before = len(df)
                df = df.drop_duplicates()
                st.session_state["df"] = df
                st.session_state["msg_success"] = f"Removed {before - len(df):,} duplicates."
                st.rerun()

        with c3:
            if st.button("Drop Rows with Missing", use_container_width=True):
                before = len(df)
                df = df.dropna()
                st.session_state["df"] = df
                st.session_state["msg_success"] = f"Dropped {before - len(df):,} rows."
                st.rerun()
    else:
        st.info("👆 Upload a dataset to get started.")


# ══════════════════════════════════════════════════════════════
#  2 ▸ AUTO EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page.startswith("🔍"):
    st.markdown('<p class="header-badge">STEP 2</p>', unsafe_allow_html=True)
    st.title("Automated Exploratory Data Analysis")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⬅️ Please upload a dataset first.")
        st.stop()

    df = st.session_state["df"]

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Statistics", "❓ Missing Values", "🔗 Correlations", "📊 Distributions"
    ])

    # ── Tab 1: Descriptive Stats ──
    with tab1:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.subheader("Quick Insights")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        i1, i2, i3 = st.columns(3)
        i1.metric("Numeric Features", len(numeric_cols))
        i2.metric("Categorical Features", len(cat_cols))
        i3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # ── Tab 2: Missing Values ──
    with tab2:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("🎉 No missing values in this dataset!")
        else:
            miss_df = pd.DataFrame({
                "Column": missing.index,
                "Missing": missing.values,
                "Percent": (missing.values / len(df) * 100).round(2),
            })
            fig = px.bar(
                miss_df, x="Column", y="Percent",
                color="Percent",
                color_continuous_scale="Reds",
                title="Missing Values by Column (%)",
                text="Percent",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#FAFAFA",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss_df, use_container_width=True, hide_index=True)

    # ── Tab 3: Correlations ──
    with tab3:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            st.info("Need at least 2 numeric columns for correlation analysis.")
        else:
            corr = numeric_df.corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Feature Correlation Matrix",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#FAFAFA",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top correlated pairs
            st.subheader("Top Correlated Feature Pairs")
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = (
                upper.stack()
                .reset_index()
                .rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "Correlation"})
                .sort_values("Correlation", key=abs, ascending=False)
                .head(10)
            )
            st.dataframe(pairs, use_container_width=True, hide_index=True)

    # ── Tab 4: Distributions ──
    with tab4:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if numeric_cols:
            st.subheader("Numeric Feature Distributions")
            sel_num = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:4])
            if sel_num:
                n_cols = min(len(sel_num), 3)
                n_rows = (len(sel_num) + n_cols - 1) // n_cols
                fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=sel_num)
                for i, col in enumerate(sel_num):
                    r, c = divmod(i, n_cols)
                    fig.add_trace(
                        go.Histogram(x=df[col], name=col, marker_color="#6C63FF", opacity=0.8),
                        row=r + 1, col=c + 1,
                    )
                fig.update_layout(
                    showlegend=False,
                    height=320 * n_rows,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#FAFAFA",
                )
                st.plotly_chart(fig, use_container_width=True)

        if cat_cols:
            st.subheader("Categorical Feature Value Counts")
            sel_cat = st.selectbox("Select a categorical column", cat_cols)
            vc = df[sel_cat].value_counts().head(20)
            fig = px.bar(
                x=vc.index.astype(str), y=vc.values,
                labels={"x": sel_cat, "y": "Count"},
                title=f"Top Values — {sel_cat}",
                color=vc.values,
                color_continuous_scale="Purples",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#FAFAFA",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  3 ▸ INTERACTIVE VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
elif page.startswith("📊"):
    st.markdown('<p class="header-badge">STEP 3</p>', unsafe_allow_html=True)
    st.title("Custom Visualization Sandbox")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⬅️ Please upload a dataset first.")
        st.stop()

    df = st.session_state["df"]
    columns = df.columns.tolist()

    # ── Controls ──
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    chart_type = ctrl1.selectbox(
        "Chart Type",
        ["Scatter Plot", "Bar Chart", "Line Chart", "Box Plot", "Histogram",
         "Violin Plot", "Pie Chart", "Sunburst"],
    )
    x_axis = ctrl2.selectbox("X-Axis", columns, index=0)

    needs_y = chart_type not in ("Histogram", "Pie Chart")
    y_axis = ctrl3.selectbox("Y-Axis", columns, index=min(1, len(columns) - 1)) if needs_y else None

    ctrl4, ctrl5 = st.columns(2)
    color_col = ctrl4.selectbox("Color (optional)", ["None"] + columns)
    color_col = None if color_col == "None" else color_col

    agg_options = ["None", "mean", "sum", "count", "median"]
    aggregation = ctrl5.selectbox("Aggregation (for Bar / Line)", agg_options)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ── Build Chart ──
    try:
        plot_df = df.copy()
        if aggregation != "None" and chart_type in ("Bar Chart", "Line Chart"):
            group_cols = [x_axis] + ([color_col] if color_col else [])
            plot_df = plot_df.groupby(group_cols, as_index=False).agg({y_axis: aggregation})

        common = dict(
            color=color_col,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Prism,
        )

        if chart_type == "Scatter Plot":
            fig = px.scatter(plot_df, x=x_axis, y=y_axis, **common, title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Bar Chart":
            fig = px.bar(plot_df, x=x_axis, y=y_axis, **common, title=f"{y_axis} by {x_axis}")
        elif chart_type == "Line Chart":
            fig = px.line(plot_df, x=x_axis, y=y_axis, **common, title=f"{y_axis} over {x_axis}")
        elif chart_type == "Box Plot":
            fig = px.box(plot_df, x=x_axis, y=y_axis, **common, title=f"Distribution of {y_axis} by {x_axis}")
        elif chart_type == "Violin Plot":
            fig = px.violin(plot_df, x=x_axis, y=y_axis, box=True, **common, title=f"Violin — {y_axis} by {x_axis}")
        elif chart_type == "Histogram":
            fig = px.histogram(plot_df, x=x_axis, color=color_col, template="plotly_dark",
                               color_discrete_sequence=px.colors.qualitative.Prism,
                               title=f"Distribution of {x_axis}", marginal="box")
        elif chart_type == "Pie Chart":
            vc = df[x_axis].value_counts().head(15)
            fig = px.pie(names=vc.index.astype(str), values=vc.values, title=f"Proportion — {x_axis}",
                         template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Prism)
        elif chart_type == "Sunburst":
            path_cols = [x_axis] + ([color_col] if color_col and color_col != x_axis else [])
            fig = px.sunburst(df, path=path_cols, template="plotly_dark",
                              color_discrete_sequence=px.colors.qualitative.Prism,
                              title=f"Sunburst — {', '.join(path_cols)}")

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            height=560,
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not render chart: {e}")


# ══════════════════════════════════════════════════════════════
#  4 ▸ TRAIN ML MODEL
# ══════════════════════════════════════════════════════════════
elif page.startswith("⚙️"):
    st.markdown('<p class="header-badge">STEP 4</p>', unsafe_allow_html=True)
    st.title("Train a Machine Learning Model")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⬅️ Please upload a dataset first.")
        st.stop()

    df = st.session_state["df"].copy()

    # ── Configuration ──
    st.markdown("### ⚡ Configuration")
    cfg1, cfg2, cfg3 = st.columns(3)

    target_col = cfg1.selectbox("🎯 Target Column", df.columns)
    task = infer_task(df[target_col])
    cfg2.info(f"Detected task: **{task.upper()}**")

    model_options = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    model_name = cfg3.selectbox("🤖 Algorithm", list(model_options.keys()))

    algo_descriptions = {
        "Random Forest": "Tree ensemble model. Great for finding complex, non-linear patterns without easily over-fitting.",
        "Gradient Boosting": "Builds trees sequentially to correct previous errors. Very powerful and accurate.",
        "XGBoost": "Extreme Gradient Boosting. High performance, highly optimized, and very fast.",
        "Logistic Regression": "Predicts probabilities using a logistic function. Strong baseline for classification.",
        "Linear Regression": "Finds the best-fitting straight line. Strong baseline for regression.",
        "Support Vector Machine": "Finds the optimal hyperplane to separate classes. Good for high-dimensional spaces."
    }
    if model_name in algo_descriptions:
        st.info(f"💡 **Description:** {algo_descriptions[model_name]}")

    adv1, adv2, adv3 = st.columns(3)
    test_size = adv1.slider("Test Size (%)", 10, 40, 20, step=5) / 100
    scale_features = adv2.checkbox("Standardize Features", value=False)
    remove_low_variance = adv3.checkbox("Remove Low Variance Features", value=False, help="Removes features that are mostly constant. Helps with faster training.")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ── Train Button ──
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Preprocessing & training … please wait"):
            try:
                # ── Prepare ──
                df_clean = df.dropna(subset=[target_col])
                y = df_clean[target_col]
                X = df_clean.drop(columns=[target_col])
                X = preprocess_features(X)

                # Encode target for classification if needed
                le_target = None
                if task == "classification" and (y.dtype == "object" or y.dtype.name == "category"):
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y)

                if remove_low_variance:
                    selector = VarianceThreshold(threshold=0.01)
                    X_selected = selector.fit_transform(X)
                    selected_cols = X.columns[selector.get_support()]
                    X = pd.DataFrame(X_selected, columns=selected_cols)
                    st.toast(f"Removed {len(selector.get_support()) - len(selected_cols)} low-variance features from dataset.")

                if scale_features:
                    scaler = StandardScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                model = model_options[model_name]()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # Persist for prediction page
                st.session_state["trained_model"] = model
                st.session_state["model_features"] = X.columns.tolist()
                st.session_state["task"] = task
                st.session_state["le_target"] = le_target
                st.session_state["scaler"] = scaler if scale_features else None

                # ── Metrics ──
                st.markdown("---")
                st.markdown("### 📋 Model Performance")

                if task == "classification":
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
                    rec = recall_score(y_test, preds, average="weighted", zero_division=0)

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Accuracy", f"{acc:.4f}")
                    mc2.metric("F1 Score", f"{f1:.4f}")
                    mc3.metric("Precision", f"{prec:.4f}")
                    mc4.metric("Recall", f"{rec:.4f}")

                    # Confusion Matrix
                    cm = confusion_matrix(y_test, preds)
                    labels = le_target.classes_.tolist() if le_target else sorted(np.unique(np.concatenate([y_test, preds])))
                    fig_cm = px.imshow(
                        cm, text_auto=True,
                        x=[str(l) for l in labels],
                        y=[str(l) for l in labels],
                        color_continuous_scale="Purples",
                        labels=dict(x="Predicted", y="Actual"),
                        title="Confusion Matrix",
                    )
                    fig_cm.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#FAFAFA",
                        height=480,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                    st.markdown("#### 📄 Classification Report")
                    report = classification_report(y_test, preds, target_names=[str(l) for l in labels])
                    st.text(report)

                else:  # Regression
                    r2 = r2_score(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    mae = mean_absolute_error(y_test, preds)

                    mr1, mr2, mr3 = st.columns(3)
                    mr1.metric("R² Score", f"{r2:.4f}")
                    mr2.metric("RMSE", f"{rmse:.4f}")
                    mr3.metric("MAE", f"{mae:.4f}")

                    # Actual vs Predicted scatter
                    fig_avp = px.scatter(
                        x=y_test, y=preds,
                        labels={"x": "Actual", "y": "Predicted"},
                        title="Actual vs Predicted",
                        template="plotly_dark",
                        opacity=0.6,
                    )
                    fig_avp.add_trace(go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode="lines", name="Perfect", line=dict(dash="dash", color="#6C63FF"),
                    ))
                    fig_avp.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#FAFAFA",
                        height=480,
                    )
                    st.plotly_chart(fig_avp, use_container_width=True)

                # ── Feature Importance ──
                if hasattr(model, "feature_importances_"):
                    st.markdown("### 🏆 Feature Importance")
                    imp = pd.DataFrame({
                        "Feature": X.columns,
                        "Importance": model.feature_importances_,
                    }).sort_values("Importance", ascending=False).head(15)

                    fig_imp = px.bar(
                        imp, x="Importance", y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale="Purples",
                        title="Top Feature Importances",
                    )
                    fig_imp.update_layout(
                        yaxis=dict(categoryorder="total ascending"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#FAFAFA",
                        height=500,
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                st.success(f"✅ **{model_name}** trained on **{len(X_train):,}** samples — ready for predictions!")
                
                # Model Download logic
                model_pkl = pickle.dumps(model)
                st.download_button(
                    label="💾 Download Trained Model (.pkl)",
                    data=model_pkl,
                    file_name=f"{model_name.replace(' ', '_').lower()}_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)


# ══════════════════════════════════════════════════════════════
#  5 ▸ PREDICT & DOWNLOAD
# ══════════════════════════════════════════════════════════════
elif page.startswith("🎯"):
    st.markdown('<p class="header-badge">STEP 5</p>', unsafe_allow_html=True)
    st.title("Generate Predictions & Download")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    if "trained_model" not in st.session_state:
        st.warning("⬅️ Please train a model first in the **Train Model** section.")
        st.stop()

    model = st.session_state["trained_model"]
    features = st.session_state["model_features"]
    task = st.session_state["task"]
    le_target = st.session_state.get("le_target")
    scaler = st.session_state.get("scaler")

    st.info(f"🤖 Active model expects **{len(features)}** features  •  Task = **{task.upper()}**")

    pred_mode = st.radio("Prediction Input", ["Use existing dataset", "Upload new file"], horizontal=True)

    pred_df = None

    if pred_mode == "Use existing dataset":
        if "df" in st.session_state:
            pred_df = st.session_state["df"].copy()
            st.dataframe(pred_df.head(10), use_container_width=True)
        else:
            st.warning("No dataset in session. Upload one first.")

    else:
        new_file = st.file_uploader("Upload new CSV / Excel for prediction", type=["csv", "xlsx", "xls"], key="pred_upload")
        if new_file:
            pred_df = load_data(new_file)
            st.dataframe(pred_df.head(10), use_container_width=True)

    if pred_df is not None and st.button("🎯 Generate Predictions", use_container_width=True):
        with st.spinner("Running predictions …"):
            try:
                X_pred = preprocess_features(pred_df.copy())
                # Align columns to training features
                for col in features:
                    if col not in X_pred.columns:
                        X_pred[col] = 0
                X_pred = X_pred[features]

                if scaler is not None:
                    X_pred = pd.DataFrame(scaler.transform(X_pred), columns=X_pred.columns)

                predictions = model.predict(X_pred)

                if le_target is not None:
                    predictions = le_target.inverse_transform(predictions)

                output_df = pred_df.copy()
                output_df["Prediction"] = predictions

                st.session_state["predictions_df"] = output_df

                st.success(f"✅ Predictions generated for **{len(output_df):,}** rows!")
                st.dataframe(output_df, use_container_width=True)

                # Distribution of predictions
                st.markdown("### 📊 Prediction Distribution")
                if task == "classification":
                    vc = output_df["Prediction"].value_counts()
                    fig = px.pie(
                        names=vc.index.astype(str), values=vc.values,
                        title="Predicted Class Distribution",
                        template="plotly_dark",
                        color_discrete_sequence=px.colors.qualitative.Prism,
                    )
                else:
                    fig = px.histogram(
                        output_df, x="Prediction",
                        title="Predicted Value Distribution",
                        template="plotly_dark",
                        color_discrete_sequence=["#6C63FF"],
                        marginal="box",
                    )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#FAFAFA",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)

    # ── Download ──
    if "predictions_df" in st.session_state:
        st.markdown("---")
        csv_data = st.session_state["predictions_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
