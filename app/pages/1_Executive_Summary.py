import streamlit as st
import plotly.express as px
from utils import load_data,sidebar_filter

df = load_data()
df = sidebar_filter(df)

st.title("Executive Summary")

total_tweets=len(df)

avg_crisis=round(df["crisis_score"].mean(),2)
dominant_topic=(df["macro_topic"].mode()[0])
dominant_sentiment=(df["sentimen_bert"].mode()[0])

if avg_crisis<0.5:
    crisis_level="Low"
elif avg_crisis<1:
    crisis_level="Medium"
else:
    crisis_level="High"

col1,col2,col3,col4,col5=st.columns(5)

col1.metric("Total Tweets",f"{total_tweets:,}")
col2.metric("Avg Crisis",f"{avg_crisis:.2f}")
col3.metric("Crisis Level",crisis_level)
col4.metric("Top Topic",dominant_topic)
col5.metric("Sentiment",dominant_sentiment)


st.divider()


topic_dist=(df["macro_topic"].value_counts().reset_index())
topic_dist.columns=["topic","count"]

fig=px.pie(
    topic_dist,
    values="count",
    names="topic",
    title="Macro Topic Distribution"
)

st.plotly_chart(fig,use_container_width=True)