import streamlit as st
import plotly.express as px
from utils import load_data,sidebar_filter

df=load_data()
df=sidebar_filter(df)

st.title("Public Insights")

col1,col2=st.columns(2)

top_topic=(df["macro_topic"].value_counts().reset_index())
top_topic.columns=["topic","count"]

fig1=px.bar(
    top_topic.head(5),
    x="topic",
    y="count",
    title="Top 5 Discussion Topics"
    )
fig1.update_layout(xaxis_title="Macro Topic",yaxis_title="Tweet Count")
fig1.update_xaxes(tickangle=-30)
col1.plotly_chart(fig1,use_container_width=True)

engagement=(
    df.groupby("macro_topic")["likeCount"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()

)

fig2=px.bar(
    engagement,
    x="macro_topic",
    y="likeCount",
    title="Average Engagement by Topic"
)
fig2.update_layout(xaxis_title="Macro Topic",yaxis_title="Average Likes")
fig2.update_xaxes(tickangle=-30)

col2.plotly_chart(fig2,use_container_width=True)

st.divider()

st.subheader("Most Engaged Tweets")

top_tweets=(df.sort_values("likeCount",ascending=False)[["text","likeCount","retweetCount"]].head(10))

st.dataframe(
    top_tweets,
    use_container_width=True
)

st.divider()

st.subheader("Representative Public Opinions")

sample_tweets=(df[["text","sentimen_bert","macro_topic"]].sample(10))

st.dataframe(
    sample_tweets,
    use_container_width=True
)