import streamlit as st
import plotly.express as px
from utils import load_data,sidebar_filter

df = load_data()
df = sidebar_filter(df)

st.title("Sentiment Analytics")

col1,col2=st.columns(2)

sentiment_dist=(df["sentimen_bert"].value_counts().reset_index())
sentiment_dist.columns=["sentiment","count"]

fig1=px.pie(
    sentiment_dist,
    values="count",
    names="sentiment",
    title="Sentiment Distribution"
)
col1.plotly_chart(fig1,use_container_width=True)

sentiment_daily=(df.groupby([df["createdAt"].dt.date,"sentimen_bert"]).size().reset_index(name="count"))

fig2=px.line(
    sentiment_daily,
    x="createdAt",
    y="count",
    color="sentimen_bert",
    title="Daily Sentiment Trend"
)
fig2.update_layout(xaxis_title="Date",yaxis_title="Tweet Count")

col2.plotly_chart(fig2, use_container_width=True)

st.divider()

topic_sentiment=(df.groupby(["macro_topic","sentimen_bert"]).size().reset_index(name="count"))

fig3=px.bar(
    topic_sentiment,
    x="macro_topic",
    y="count",
    color="sentimen_bert",
    title="Sentiment by Macro Topic"
    )
fig3.update_layout(xaxis_title="Macro Topic",yaxis_title="Tweet Count")
fig3.update_xaxes(tickangle=-30)

st.plotly_chart(
    fig3,
    use_container_width=True
)