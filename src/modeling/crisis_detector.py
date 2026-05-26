import pandas as pd


class CrisisDetector:

    def __init__(self):

        self.topic_mapping={

            "Global Economic Crisis":[8,6],

            "Macroeconomic Crisis & Government Response":[
                13,12,4,0,10,21
            ],

            "Rupiah Exchange Dynamics":[
                18,20,5,3,15
            ],

            "Food Price & Inflation Control":[
                11,24,19
            ],

            "Inflation & Public Economic Impact":[
                22,9,26,7,2,1,17,14,16
            ],

            "Financial Market & Digital Assets":[
                25,23
            ]

        }

        self.weights={

            "Macroeconomic Crisis & Government Response":1.5,
            "Inflation & Public Economic Impact":1.4,
            "Rupiah Exchange Dynamics":1.3,
            "Food Price & Inflation Control":1.2,
            "Global Economic Crisis":1.1,
            "Financial Market & Digital Assets":1.0,
            "Outlier":0.7

        }

        self.sentiment_score={

            "negative":1,
            "neutral":0.5,
            "positive":0.2

        }


    def merge_data(self,df_sentiment,df_topics):

        df=pd.merge(
            df_sentiment,
            df_topics[["id","topic"]],
            on="id",
            how="inner"
        )

        return df


    def map_topic(self,df):

        topic_to_macro={}

        for macro,topics in self.topic_mapping.items():

            for t in topics:
                topic_to_macro[t]=macro

        df["macro_topic"]=df[
            "topic"
        ].map(
            topic_to_macro
        )

        df["macro_topic"]=df[
            "macro_topic"
        ].fillna(
            "Outlier"
        )

        return df


    def topic_weight(self,df):

        df["topic_weight"]=df[
            "macro_topic"
        ].map(
            self.weights
        )

        return df


    def sentiment_weight(self,df):

        df["sentiment_score"]=df[
            "sentimen_bert"
        ].map(
            self.sentiment_score
        )

        return df


    def calculate_crisis(self,df):

        df["crisis_score"]=(

            df["sentiment_score"]
            *
            df["topic_weight"]
            *
            df["confidence"]

        )

        return df


    def daily_crisis(self,df):

        df["createdAt"]=pd.to_datetime(
            df["createdAt"]
        )

        daily_crisis=(

            df.groupby(
                df["createdAt"].dt.date
            )["crisis_score"]

            .mean()

            .reset_index()

        )

        daily_crisis.columns=[
            "date",
            "avg_crisis_score"
        ]

        daily_crisis["crisis_level"]=(
            daily_crisis[
                "avg_crisis_score"
            ].apply(
                self.crisis_level
            )
        )

        return daily_crisis


    def crisis_level(self,score):

        if score<0.5:
            return "Low"

        elif score<1:
            return "Medium"

        return "High"