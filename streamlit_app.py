import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")


# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_df():
    df = pd.read_csv("tourism_merged.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


# -------------------------
# Train regression inside app
# -------------------------
@st.cache_resource
def train_regression(df: pd.DataFrame):
    features = [
        "VisitYear", "VisitMonth", "VisitQuarter", "VisitSeason",
        "UserContinentName", "UserRegionName", "UserCountryName",
        "AttractionTypeName"
    ]
    target = "Rating"

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[features + [target]].dropna().copy()
    X = data[features]
    y = data[target].astype(float)

    cat_cols = ["VisitSeason", "UserContinentName", "UserRegionName", "UserCountryName", "AttractionTypeName"]
    num_cols = ["VisitYear", "VisitMonth", "VisitQuarter"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    reg = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline([("preprocess", pre), ("model", reg)])
    pipe.fit(X_train, y_train)

    r2 = r2_score(y_test, pipe.predict(X_test))
    return pipe, r2


# -------------------------
# Train classification inside app (replaces joblib .pkl)
# -------------------------
@st.cache_resource
def train_classifier(df: pd.DataFrame):
    features = [
        "VisitYear", "VisitMonth", "VisitQuarter", "VisitSeason",
        "UserContinentName", "AttractionTypeName",
        "user_avg_rating", "user_rating_count",
        "attraction_avg_rating", "attraction_rating_count"
    ]
    target = "VisitModeName"

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required classification columns: {missing}")

    data = df[features + [target]].dropna().copy()
    X = data[features]
    y = data[target].astype(str)

    cat_cols = ["VisitSeason", "UserContinentName", "AttractionTypeName"]
    num_cols = [
        "VisitYear", "VisitMonth", "VisitQuarter",
        "user_avg_rating", "user_rating_count",
        "attraction_avg_rating", "attraction_rating_count"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    clf = HistGradientBoostingClassifier(random_state=42)

    pipe = Pipeline([("preprocess", pre), ("model", clf)])
    pipe.fit(X_train, y_train)

    # optional: quick metric in logs
    acc = pipe.score(X_test, y_test)
    return pipe, acc


# -------------------------
# Recommender (collab + popular)
# -------------------------
@st.cache_resource
def build_recommender(df: pd.DataFrame):
    required = ["UserId", "AttractionId", "Rating", "AttractionTypeName", "Attraction", "AttractionCityId"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required recommender columns: {missing}")

    rec = df[required].dropna().copy()
    rec = rec.groupby(["UserId", "AttractionId"], as_index=False).agg({
        "Rating": "mean",
        "AttractionTypeName": "first",
        "Attraction": "first",
        "AttractionCityId": "first"
    })

    train_rows = []
    for u, g in rec.groupby("UserId"):
        if len(g) < 5:
            train_rows.append(g)
        else:
            tr, _ = train_test_split(g, test_size=0.2, random_state=42)
            train_rows.append(tr)

    train_df = pd.concat(train_rows, ignore_index=True)

    user_item = train_df.pivot_table(
        index="UserId",
        columns="AttractionId",
        values="Rating",
        fill_value=0
    )

    if user_item.shape[0] == 0 or user_item.shape[1] == 0:
        sim_df = pd.DataFrame()
    else:
        sim = cosine_similarity(user_item.T)
        sim_df = pd.DataFrame(sim, index=user_item.columns, columns=user_item.columns)

    meta = train_df.drop_duplicates("AttractionId")[["AttractionId", "Attraction", "AttractionTypeName"]].set_index("AttractionId")

    pop = train_df.groupby("AttractionId").agg(
        avg_rating=("Rating", "mean"),
        n=("Rating", "count")
    ).reset_index()
    pop["score"] = pop["avg_rating"] * np.log1p(pop["n"])
    pop = pop.sort_values("score", ascending=False)

    return user_item, sim_df, meta, pop


def recommend_collab(user_id, user_item, sim_df, meta, top_n=10):
    if user_item is None or user_item.empty or sim_df is None or sim_df.empty:
        return pd.DataFrame()

    if user_id not in user_item.index:
        return pd.DataFrame()

    r = user_item.loc[user_id]
    rated = r[r > 0].index.tolist()
    if not rated:
        return pd.DataFrame()

    scores = pd.Series(0.0, index=user_item.columns)
    for it in rated:
        if it in sim_df.columns:
            scores += sim_df[it] * r[it]

    scores = scores.drop(index=rated, errors="ignore")
    top = scores.sort_values(ascending=False).head(top_n)

    out = pd.DataFrame({"AttractionId": top.index, "Score": top.values})
    out = out.join(meta, on="AttractionId")
    return out[["AttractionId", "Attraction", "AttractionTypeName", "Score"]]


def recommend_popular(pop, meta, top_n=10):
    top = pop.head(top_n).copy()
    top = top.join(meta, on="AttractionId")
    cols = ["AttractionId", "Attraction", "AttractionTypeName", "avg_rating", "n", "score"]
    return top[cols]


# -------------------------
# App
# -------------------------
st.title("Tourism Experience Analytics")

df = load_df()

# Train models in-app
try:
    reg_model, reg_r2 = train_regression(df)
except Exception as e:
    st.error("Regression training failed.")
    st.code(str(e))
    st.stop()

try:
    clf_model, clf_acc = train_classifier(df)
except Exception as e:
    st.error("Classification training failed.")
    st.code(str(e))
    st.stop()

try:
    user_item, sim_df, meta, pop = build_recommender(df)
except Exception as e:
    st.error("Recommender build failed.")
    st.code(str(e))
    st.stop()

st.caption(f"Regression (in-app) R² ≈ {reg_r2:.3f} | Classification (in-app) Acc ≈ {clf_acc:.3f}")

# Sidebar inputs
st.sidebar.header("Inputs")

min_uid = int(df["UserId"].min())
max_uid = int(df["UserId"].max())
user_id = st.sidebar.number_input("UserId", min_value=min_uid, max_value=max_uid, value=min_uid, step=1)

visit_year_options = sorted(df["VisitYear"].dropna().unique().tolist())
visit_year = st.sidebar.selectbox("VisitYear", visit_year_options, index=0)

visit_month = st.sidebar.selectbox("VisitMonth", list(range(1, 13)), index=0)

visit_quarter = int((visit_month - 1) // 3 + 1)
visit_season = (
    "Winter" if visit_month in [12, 1, 2] else
    "Spring" if visit_month in [3, 4, 5] else
    "Summer" if visit_month in [6, 7, 8] else
    "Autumn"
)

cont_options = sorted(df["UserContinentName"].dropna().unique().tolist())
reg_options = sorted(df["UserRegionName"].dropna().unique().tolist())
country_options = sorted(df["UserCountryName"].dropna().unique().tolist())
type_options = sorted(df["AttractionTypeName"].dropna().unique().tolist())

user_cont = st.sidebar.selectbox("UserContinentName", cont_options, index=0)
user_region = st.sidebar.selectbox("UserRegionName", reg_options, index=0)
user_country = st.sidebar.selectbox("UserCountryName", country_options, index=0)
attr_type = st.sidebar.selectbox("AttractionTypeName", type_options, index=0)

top_n = st.sidebar.slider("Top-N Recommendations", 5, 20, 10)

# Regression input
X_reg = pd.DataFrame([{
    "VisitYear": int(visit_year),
    "VisitMonth": int(visit_month),
    "VisitQuarter": int(visit_quarter),
    "VisitSeason": str(visit_season),
    "UserContinentName": str(user_cont),
    "UserRegionName": str(user_region),
    "UserCountryName": str(user_country),
    "AttractionTypeName": str(attr_type),
}])

# Classification input feature creation
u_stats = df.loc[df["UserId"] == user_id, "Rating"]
user_avg = float(u_stats.mean()) if len(u_stats) else float(df["Rating"].mean())
user_cnt = float(u_stats.count()) if len(u_stats) else 0.0

type_df = df[df["AttractionTypeName"] == attr_type]
attr_avg = float(type_df["Rating"].mean()) if len(type_df) else float(df["Rating"].mean())
attr_cnt = float(len(type_df)) if len(type_df) else float(df.groupby("AttractionId").size().mean())

X_clf = pd.DataFrame([{
    "VisitYear": int(visit_year),
    "VisitMonth": int(visit_month),
    "VisitQuarter": int(visit_quarter),
    "VisitSeason": str(visit_season),
    "UserContinentName": str(user_cont),
    "AttractionTypeName": str(attr_type),
    "user_avg_rating": user_avg,
    "user_rating_count": user_cnt,
    "attraction_avg_rating": attr_avg,
    "attraction_rating_count": attr_cnt,
}])

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Predicted Visit Mode")
    st.success(clf_model.predict(X_clf)[0])

with c2:
    st.subheader("Predicted Rating")
    st.info(f"{float(reg_model.predict(X_reg)[0]):.2f} / 5")

with c3:
    st.subheader("Recommendations")
    recs = recommend_collab(int(user_id), user_item, sim_df, meta, top_n)
    if recs.empty:
        st.warning("No history for this user → Popular attractions")
        st.dataframe(recommend_popular(pop, meta, top_n), use_container_width=True)
    else:
        st.dataframe(recs, use_container_width=True)

with st.expander("Debug: Inputs"):
    st.write("X_clf")
    st.dataframe(X_clf)
    st.write("X_reg")
    st.dataframe(X_reg)
