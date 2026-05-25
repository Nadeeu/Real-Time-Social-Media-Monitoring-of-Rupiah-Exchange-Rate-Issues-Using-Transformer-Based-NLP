import streamlit as st
import plotly.express as px
from utils import load_data,sidebar_filter

df=load_data()
df=sidebar_filter(df)

st.title("Crisis Monitoring")

daily_crisis=(df.groupby(df["createdAt"].dt.date)["crisis_score"].mean().reset_index())
daily_crisis.columns=["date","crisis_score"]

avg_crisis=round(df["crisis_score"].mean(),2)

if avg_crisis<0.5:
    crisis_level="Low"
elif avg_crisis<1:
    crisis_level="Medium"
else:
    crisis_level="High"

col1,col2,col3=st.columns(3)

col1.metric("Current Crisis Score",f"{avg_crisis:.2f}")
col2.metric("Crisis Level",crisis_level)
col3.metric("Highest Score",f"{daily_crisis['crisis_score'].max():.2f}")
st.divider()

fig1=px.line(
    daily_crisis,
    x="date",
    y="crisis_score",
    title="Daily Crisis Trend"
)
fig1.update_layout(xaxis_title="Date",yaxis_title="Crisis Score")

st.plotly_chart(fig1,use_container_width=True)

trigger_topic=(df.groupby("macro_topic")["crisis_score"].mean().sort_values(ascending=False).reset_index())

fig2=px.bar(
    trigger_topic,
    x="macro_topic",
    y="crisis_score",
    title="Top Crisis Trigger Topics"
)
fig2.update_layout(xaxis_title="Macro Topic",yaxis_title="Average Crisis Score")
fig2.update_xaxes(tickangle=-30)

st.plotly_chart(fig2,use_container_width=True)