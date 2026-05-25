import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent

DATA_PATH=(
    BASE_DIR/
    "data"/
    "processed"/
    "final_data.csv"
)

@st.cache_data
def load_data():

    df=pd.read_csv(DATA_PATH)
    df["createdAt"]=pd.to_datetime(df["createdAt"])

    return df


def sidebar_filter(df):
    st.sidebar.header("Filter")
    
    if st.sidebar.button("Reset Filter"):
        st.cache_data.clear()
        st.rerun()
    
    selected_topic=st.sidebar.multiselect(
        "Macro Topic",
        options=sorted(df["macro_topic"].dropna().unique()),
        default=sorted(df["macro_topic"].dropna().unique())
        )

    selected_sentiment=st.sidebar.multiselect(
        "Sentiment",
        options=df["sentimen_bert"].dropna().unique(),
        default=df["sentimen_bert"].dropna().unique()
        )

    min_date=df["createdAt"].min()
    max_date=df["createdAt"].max()

    selected_date=st.sidebar.date_input(
        "Date Range",
        [min_date,max_date]
        )

    df=df[
        (df["macro_topic"].isin(selected_topic))
        &
        (df["sentimen_bert"].isin(selected_sentiment))
        &
        (
            df["createdAt"]
            .dt.date
            .between(
                selected_date[0],
                selected_date[1]
            )
        )
    ]

    return df