import streamlit as st
import plotly.express as px
from utils import load_data,sidebar_filter

df=load_data()
df=sidebar_filter(df)

st.title("Topic Analytics")

col1,col2=st.columns(2)

topic_dist=(df["macro_topic"].value_counts().reset_index())
topic_dist.columns=["macro_topic","count"]

fig1=px.bar(
    topic_dist,
    x="macro_topic",
    y="count",
    title="Macro Topic Distribution"
)
fig1.update_layout(xaxis_title="Macro Topic",yaxis_title="Tweet Count")
fig1.update_xaxes(tickangle=-30)

col1.plotly_chart(fig1,use_container_width=True)

topic_crisis=(
    df.groupby("macro_topic")["crisis_score"]
    .mean()
    .reset_index()
)

fig2=px.bar(
    topic_crisis,
    x="macro_topic",
    y="crisis_score",
    title="Average Crisis Score by Topic"
)
fig2.update_layout(xaxis_title="Macro Topic",yaxis_title="Average Crisis Score")
fig2.update_xaxes(tickangle=-30)

col2.plotly_chart(fig2,use_container_width=True)

st.divider()

topic_summary=(

    df.groupby("macro_topic")
    .agg(
        tweet_count=("id","count"),
        avg_crisis=("crisis_score","mean"),
        avg_likes=("likeCount","mean")
    )
    .reset_index()

)

st.subheader("Topic Summary")

st.dataframe(
    topic_summary,
    use_container_width=True
)