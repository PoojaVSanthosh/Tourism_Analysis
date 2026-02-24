import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")


# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_df():
    df = pd.read_csv("tourism_merged.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


df = load_df()

st.title("Tourism Experience Analytics")


# =====================================================
# Simple Rating Prediction (Mean-Based)
# =====================================================
def predict_rating(user_id, attraction_type):
    user_ratings = df[df["UserId"] == user_id]["Rating"]

    if len(user_ratings) > 0:
        return float(user_ratings.mean())

    return float(df["Rating"].mean())


# =====================================================
# Simple Visit Mode Prediction (Mode-Based)
# =====================================================
def predict_visit_mode(user_id):
    user_modes = df[df["UserId"] == user_id]["VisitModeName"]

    if len(user_modes) > 0:
        return user_modes.mode()[0]

    return df["VisitModeName"].mode()[0]


# =====================================================
# Simple Popular Recommendation
# =====================================================
def recommend_popular(top_n=10):
    pop = df.groupby("AttractionId").agg(
        avg_rating=("Rating", "mean"),
        n=("Rating", "count"),
        Attraction=("Attraction", "first"),
        AttractionTypeName=("AttractionTypeName", "first")
    ).reset_index()

    pop["score"] = pop["avg_rating"] * np.log1p(pop["n"])
    pop = pop.sort_values("score", ascending=False)

    return pop.head(top_n)[
        ["AttractionId", "Attraction", "AttractionTypeName", "avg_rating", "n"]
    ]


# =====================================================
# Sidebar Inputs
# =====================================================
st.sidebar.header("Inputs")

min_uid = int(df["UserId"].min())
max_uid = int(df["UserId"].max())

user_id = st.sidebar.number_input(
    "UserId", min_value=min_uid, max_value=max_uid, value=min_uid
)

attr_types = sorted(df["AttractionTypeName"].dropna().unique())
attr_type = st.sidebar.selectbox("AttractionTypeName", attr_types)

top_n = st.sidebar.slider("Top Recommendations", 5, 20, 10)


# =====================================================
# Predictions
# =====================================================
pred_rating = predict_rating(user_id, attr_type)
pred_mode = predict_visit_mode(user_id)
recommendations = recommend_popular(top_n)


c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Predicted Visit Mode")
    st.success(pred_mode)

with c2:
    st.subheader("Predicted Rating")
    st.info(f"{pred_rating:.2f} / 5")

with c3:
    st.subheader("Top Attractions")
    st.dataframe(recommendations, use_container_width=True)
