# ============================================================
# DIABETES PREDICTION & CSV ANALYSIS WEB APP
# InternPe Internship Project
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

h1 {
    text-align: center;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}

div[data-testid="stMetric"] {
    border: 1px solid #dddddd;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Diabetes Prediction & Data Analysis")

st.markdown(
    '<p class="subtitle">'
    'Upload a CSV dataset and perform complete data analysis '
    'and machine-learning prediction.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Navigation")

st.sidebar.info(
    """
    This application performs:

    • CSV Upload
    • Dataset Analysis
    • Missing Value Analysis
    • Statistical Analysis
    • Data Visualization
    • Correlation Analysis
    • Machine Learning
    • Model Evaluation
    • Diabetes Prediction
    """
)


# ============================================================
# CSV UPLOAD
# ============================================================

st.header("📂 Upload Your CSV Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# ============================================================
# WAIT FOR FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Please upload a CSV file to begin the analysis."
    )

    st.markdown("""
    ### How to use

    1. Click **Browse files**
    2. Select your diabetes CSV file
    3. Wait for the dataset to load
    4. Explore the analysis
    5. Train the machine-learning model
    6. Enter patient details to make a prediction

    ### Expected diabetes dataset

    Your dataset should preferably contain columns such as:

    - Pregnancies
    - Glucose
    - BloodPressure
    - SkinThickness
    - Insulin
    - BMI
    - DiabetesPedigreeFunction
    - Age
    - Outcome

    **Outcome:**
    - 0 = No Diabetes
    - 1 = Diabetes
    """)

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(
        f"❌ Error reading CSV file: {error}"
    )

    st.stop()


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    f"✅ Successfully loaded: {uploaded_file.name}"
)


# ============================================================
# DATASET PREVIEW
# ============================================================

st.header("📋 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Rows",
        df.shape[0]
    )


with col2:

    st.metric(
        "Total Columns",
        df.shape[1]
    )


with col3:

    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


with col4:

    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )


# ============================================================
# COLUMN INFORMATION
# ============================================================

st.subheader("📝 Column Names")

st.write(
    list(df.columns)
)


# ============================================================
# DATA TYPES
# ============================================================

st.subheader("🔤 Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

st.dataframe(
    dtype_df,
    use_container_width=True
)


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

st.header("📈 Statistical Summary")

try:

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

except Exception:

    st.warning(
        "Statistical summary is unavailable for this dataset."
    )


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

st.header("🔍 Missing Value Analysis")

missing_values = df.isnull().sum()

missing_df = pd.DataFrame({
    "Column": missing_values.index,
    "Missing Values": missing_values.values
})

missing_df = missing_df[
    missing_df["Missing Values"] > 0
]


if missing_df.empty:

    st.success(
        "✅ No missing values were found."
    )

else:

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=missing_df,
        x="Column",
        y="Missing Values",
        ax=ax
    )

    ax.set_title(
        "Missing Values by Column"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

st.header("🔄 Duplicate Row Analysis")

duplicate_count = int(
    df.duplicated().sum()
)


if duplicate_count == 0:

    st.success(
        "✅ No duplicate rows found."
    )

else:

    st.warning(
        f"⚠️ {duplicate_count} duplicate rows found."
    )


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


# ============================================================
# NUMERICAL DATA DISTRIBUTION
# ============================================================

if len(numeric_columns) > 0:

    st.header("📊 Data Distribution")

    selected_column = st.selectbox(
        "Select a numerical column:",
        numeric_columns
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.histplot(
        df[selected_column].dropna(),
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_column}"
    )

    ax.set_xlabel(
        selected_column
    )

    ax.set_ylabel(
        "Frequency"
    )

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

if len(numeric_columns) >= 2:

    st.header("🔥 Correlation Heatmap")

    correlation = df[
        numeric_columns
    ].corr()

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation Between Numerical Features"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# DETECT OUTCOME COLUMN
# ============================================================

outcome_column = None

possible_outcome_names = [
    "Outcome",
    "outcome",
    "Diabetes",
    "diabetes",
    "Target",
    "target",
    "Label",
    "label"
]


for column in possible_outcome_names:

    if column in df.columns:

        outcome_column = column

        break


# ============================================================
# DIABETES OUTCOME ANALYSIS
# ============================================================

if outcome_column is not None:

    st.header("🩺 Diabetes Outcome Analysis")

    outcome_counts = df[
        outcome_column
    ].value_counts()

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Outcome Counts")

        st.dataframe(
            outcome_counts,
            use_container_width=True
        )


    with col2:

        st.subheader("Outcome Distribution")

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.countplot(
            x=outcome_column,
            data=df,
            ax=ax
        )

        ax.set_title(
            "Diabetes Outcome Distribution"
        )

        ax.set_xlabel(
            "Outcome"
        )

        ax.set_ylabel(
            "Number of Patients"
        )

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# MACHINE LEARNING SECTION
# ============================================================

st.header("🤖 Machine Learning")


if outcome_column is None:

    st.warning(
        """
        ⚠️ No target/outcome column was detected.

        For diabetes prediction, your CSV should contain
        an Outcome column such as:

        0 = No Diabetes
        1 = Diabetes
        """
    )

else:

    # --------------------------------------------------------
    # SELECT NUMERICAL FEATURES
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in numeric_columns
        if column != outcome_column
    ]


    if len(feature_columns) == 0:

        st.error(
            "❌ No numerical features are available."
        )

    else:

        X = df[
            feature_columns
        ].copy()

        y = df[
            outcome_column
        ].copy()


        # ----------------------------------------------------
        # CONVERT FEATURES TO NUMERIC
        # ----------------------------------------------------

        for column in X.columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )


        # ----------------------------------------------------
        # CONVERT TARGET TO NUMERIC
        # ----------------------------------------------------

        y = pd.to_numeric(
            y,
            errors="coerce"
        )


        # ----------------------------------------------------
        # REMOVE INVALID TARGET VALUES
        # ----------------------------------------------------

        valid_rows = y.notna()

        X = X.loc[
            valid_rows
        ]

        y = y.loc[
            valid_rows
        ]


        # ----------------------------------------------------
        # REPLACE INFINITY
        # ----------------------------------------------------

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )


        # ----------------------------------------------------
        # REMOVE EMPTY COLUMNS
        # ----------------------------------------------------

        X = X.dropna(
            axis=1,
            how="all"
        )


        # ----------------------------------------------------
        # FILL MISSING VALUES
        # ----------------------------------------------------

        X = X.fillna(
            X.median()
        )


        # ----------------------------------------------------
        # CHECK TARGET CLASSES
        # ----------------------------------------------------

        unique_classes = sorted(
            y.unique()
        )


        if len(unique_classes) != 2:

            st.error(
                f"""
                ❌ The target column must contain exactly
                two classes.

                Detected classes:
                {unique_classes}
                """
            )

        else:

            # =================================================
            # TRAIN / TEST SPLIT
            # =================================================

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )


            # =================================================
            # FEATURE SCALING
            # =================================================

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(
                X_train
            )

            X_test_scaled = scaler.transform(
                X_test
            )


            # =================================================
            # TRAIN MODEL
            # =================================================

            model = LogisticRegression(
                max_iter=1000
            )

            model.fit(
                X_train_scaled,
                y_train
            )


            # =================================================
            # PREDICTIONS
            # =================================================

            y_pred = model.predict(
                X_test_scaled
            )

            y_probability = model.predict_proba(
                X_test_scaled
            )[:, 1]


            # =================================================
            # MODEL ACCURACY
            # =================================================

            accuracy = accuracy_score(
                y_test,
                y_pred
            )


            # =================================================
            # ROC AUC
            # =================================================

            try:

                roc_auc = roc_auc_score(
                    y_test,
                    y_probability
                )

            except Exception:

                roc_auc = 0


            # =================================================
            # MODEL PERFORMANCE
            # =================================================

            st.subheader(
                "📊 Model Performance"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "ROC-AUC",
                    f"{roc_auc:.3f}"
                )


            with col3:

                st.metric(
                    "Training Samples",
                    len(X_train)
                )


            # =================================================
            # CONFUSION MATRIX
            # =================================================

            st.subheader(
                "🎯 Confusion Matrix"
            )


            cm = confusion_matrix(
                y_test,
                y_pred
            )


            fig, ax = plt.subplots(
                figsize=(6, 5)
            )


            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )


            ax.set_xlabel(
                "Predicted"
            )

            ax.set_ylabel(
                "Actual"
            )

            ax.set_title(
                "Confusion Matrix"
            )


            st.pyplot(fig)

            plt.close(fig)


            # =================================================
            # CLASSIFICATION REPORT
            # =================================================

            st.subheader(
                "📋 Classification Report"
            )


            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0
            )


            report_df = pd.DataFrame(
                report
            ).transpose()


            st.dataframe(
                report_df.round(3),
                use_container_width=True
            )


            # =================================================
            # ROC CURVE
            # =================================================

            st.subheader(
                "📈 ROC Curve"
            )


            try:

                fpr, tpr, thresholds = roc_curve(
                    y_test,
                    y_probability
                )


                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )


                ax.plot(
                    fpr,
                    tpr,
                    label=f"AUC = {roc_auc:.3f}"
                )


                ax.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--"
                )


                ax.set_xlabel(
                    "False Positive Rate"
                )

                ax.set_ylabel(
                    "True Positive Rate"
                )

                ax.set_title(
                    "ROC Curve"
                )

                ax.legend()


                st.pyplot(fig)

                plt.close(fig)

            except Exception:

                st.warning(
                    "ROC curve could not be generated."
                )


            # =================================================
            # FEATURE IMPORTANCE
            # =================================================

            st.subheader(
                "🔎 Feature Importance"
            )


            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": model.coef_[0]
            })


            importance_df[
                "Absolute Importance"
            ] = importance_df[
                "Coefficient"
            ].abs()


            importance_df = importance_df.sort_values(
                by="Absolute Importance",
                ascending=False
            )


            st.dataframe(
                importance_df,
                use_container_width=True
            )


            fig, ax = plt.subplots(
                figsize=(10, 6)
            )


            sns.barplot(
                data=importance_df,
                x="Coefficient",
                y="Feature",
                ax=ax
            )


            ax.set_title(
                "Feature Importance - Logistic Regression"
            )


            st.pyplot(fig)

            plt.close(fig)


            # =================================================
            # NEW PATIENT PREDICTION
            # =================================================

            st.header(
                "🧑‍⚕️ Diabetes Prediction"
            )


            st.write(
                "Enter patient information below "
                "to generate a model prediction."
            )


            user_values = {}


            prediction_columns = st.columns(2)


            for index, feature in enumerate(
                X.columns
            ):

                with prediction_columns[
                    index % 2
                ]:

                    median_value = float(
                        X[feature].median()
                    )


                    user_values[feature] = st.number_input(
                        feature,
                        value=median_value
                    )


            st.divider()


            if st.button(
                "🔮 Predict Diabetes",
                type="primary"
            ):

                new_patient = pd.DataFrame(
                    [user_values]
                )


                new_patient_scaled = scaler.transform(
                    new_patient
                )


                prediction = model.predict(
                    new_patient_scaled
                )[0]


                probability = model.predict_proba(
                    new_patient_scaled
                )[0][1]


                st.subheader(
                    "Prediction Result"
                )


                result_col1, result_col2 = st.columns(2)


                with result_col1:

                    st.metric(
                        "Diabetes Probability",
                        f"{probability * 100:.2f}%"
                    )


                with result_col2:

                    if prediction == 1:

                        st.error(
                            "⚠️ Prediction: Diabetes"
                        )

                    else:

                        st.success(
                            "✅ Prediction: No Diabetes"
                        )


                # --------------------------------------------
                # PROBABILITY BAR
                # --------------------------------------------

                st.progress(
                    float(probability)
                )


                st.caption(
                    "⚠️ This is an educational machine-learning "
                    "prediction and is not a medical diagnosis."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <p><b>Diabetes Prediction & Analysis</b></p>
        <p>Machine Learning Project | InternPe</p>
    </div>
    """,
    unsafe_allow_html=True
)