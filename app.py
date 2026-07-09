import os
import json
import joblib
import pandas as pd
import streamlit as st

# ---------------------- PAGE CONFIG ---------------------- #

st.set_page_config(
    page_title="Loan Rejection Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- CSS ---------------------- #

st.markdown(
    """
<style>

footer{
visibility:hidden;
}

.main > div{
padding-top:1rem;
}

.stButton>button{
width:100%;
border-radius:8px;
height:3em;
font-weight:bold;
}

div[data-testid="stMetric"]{
border:1px solid #dddddd;
padding:12px;
border-radius:10px;
}

</style>
""",
    unsafe_allow_html=True
)

# ---------------------- PATHS ---------------------- #

MODEL_DIR = "models"
DEPLOYMENT_DIR = "deployment"

# ---------------------- LOAD ---------------------- #

@st.cache_resource
def load_resources():

    try:

        model = joblib.load(
            os.path.join(MODEL_DIR, "best_model.pkl")
        )

        feature_info = joblib.load(
            os.path.join(MODEL_DIR, "feature_columns.pkl")
        )

        with open(
            os.path.join(
                DEPLOYMENT_DIR,
                "metadata.json"
            ),
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        with open(
            os.path.join(
                DEPLOYMENT_DIR,
                "deployment_config.json"
            ),
            "r",
            encoding="utf-8"
        ) as f:

            deployment = json.load(f)

        return (
            model,
            feature_info,
            metadata,
            deployment
        )

    except Exception as e:

        st.error(
            f"Deployment Error\n\n{e}"
        )

        st.stop()


(
    model,
    feature_info,
    metadata,
    deployment
) = load_resources()

# ---------------------- FEATURES ---------------------- #

if isinstance(feature_info, dict):

    feature_columns = feature_info.get(
        "original_features",
        feature_info.get(
            "feature_columns",
            []
        )
    )

    numeric_features = feature_info.get(
        "numeric_features",
        feature_columns
    )

else:

    feature_columns = list(feature_info)

    numeric_features = feature_columns

# ---------------------- SIDEBAR ---------------------- #

st.sidebar.title("🏦 Loan Prediction")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📝 Loan Prediction",

        "📊 Dataset",

        "🤖 Model",

        "👨‍💻 Developer"

    ]

)

# ---------------------- HOME ---------------------- #

if page == "🏠 Home":

    st.title(
        "🏦 Loan Rejection Prediction System"
    )

    st.write(
        """
Predict whether a loan application
is likely to be approved or rejected
using a trained Machine Learning Pipeline.
"""
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(

            "Training Rows",

            metadata.get(
                "training_rows",
                "-"
            )

        )

    with c2:

        st.metric(

            "Testing Rows",

            metadata.get(
                "testing_rows",
                "-"
            )

        )

    with c3:

        acc = metadata.get(
            "accuracy",
            None
        )

        if acc is not None:

            st.metric(
                "Accuracy",
                f"{acc:.2%}"
            )

    st.info(
        "Use the sidebar to navigate through the application."
    )

# ---------------------- DATASET ---------------------- #

elif page == "📊 Dataset":

    st.title("📊 Dataset Information")

    st.metric(
        "Dataset",
        metadata.get(
            "dataset_name",
            "-"
        )
    )

    st.metric(
        "Features",
        metadata.get(
            "number_of_features",
            len(feature_columns)
        )
    )

    st.json(metadata)
    # ---------------------- MODEL ---------------------- #

elif page == "🤖 Model":

    st.title("🤖 Model Information")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Best Model",
            metadata.get(
                "best_model",
                "-"
            )
        )

        st.metric(
            "Accuracy",
            f"{metadata.get('accuracy',0):.2%}"
        )

        st.metric(
            "Precision",
            f"{metadata.get('precision',0):.2%}"
        )

    with c2:

        st.metric(
            "Recall",
            f"{metadata.get('recall',0):.2%}"
        )

        st.metric(
            "F1 Score",
            f"{metadata.get('f1',0):.2%}"
        )

        st.metric(
            "ROC AUC",
            f"{metadata.get('roc_auc',0):.2%}"
        )

    st.markdown("---")

    st.subheader("Pipeline")

    st.write("""
The deployed model is a **complete Scikit-learn Pipeline**.

Pipeline Components

• Median Imputation

• Standard Scaling

• ColumnTransformer

• Machine Learning Classifier

The application loads the saved pipeline directly.

No preprocessing or retraining occurs during deployment.
""")

# ---------------------- PREDICTION ---------------------- #

elif page == "📝 Loan Prediction":

    st.title("📝 Loan Prediction")

    st.info(
        "Enter values for the required features and click Predict."
    )

    user_input = {}

    columns = st.columns(3)

    for index, feature in enumerate(feature_columns):

        with columns[index % 3]:

            user_input[feature] = st.number_input(

                label=feature,

                value=0.0,

                step=1.0,

                format="%.4f"

            )

    if st.button("Predict Loan Status"):

        with st.spinner("Running prediction..."):

            try:

                input_df = pd.DataFrame(
                    [user_input]
                )

                prediction = model.predict(
                    input_df
                )[0]

                probability = None

                if hasattr(model, "predict_proba"):

                    probability = model.predict_proba(
                        input_df
                    )[0]

                st.markdown("---")

                if prediction == 1:

                    st.success(
                        "✅ Prediction : LOAN APPROVED"
                    )

                    if probability is not None:

                        st.metric(
                            "Approval Probability",
                            f"{probability[1]:.2%}"
                        )

                else:

                    st.error(
                        "❌ Prediction : LOAN REJECTED"
                    )

                    if probability is not None:

                        st.metric(
                            "Rejection Probability",
                            f"{probability[0]:.2%}"
                        )

            except Exception as e:

                st.error(
                    f"Prediction failed.\n\n{e}"
                )

# ---------------------- DEVELOPER ---------------------- #

elif page == "👨‍💻 Developer":

    st.title("👨‍💻 Developer")

    st.write("Loan Rejection Prediction System")

    st.write("Summer School Assignment 2")

    st.write("Developed using")

    st.write("- Python")

    st.write("- Scikit-learn")

    st.write("- Streamlit")

    st.write("- Google Colab")

    st.write("- Joblib")

    st.write("- Pandas")

    st.write("- NumPy")

    st.markdown("---")

    st.caption(
        "Deployment Optimized for Streamlit Community Cloud"
    )

# ---------------------- FOOTER ---------------------- #

st.markdown("---")

st.caption(
    "© 2026 Loan Rejection Prediction System"
)
