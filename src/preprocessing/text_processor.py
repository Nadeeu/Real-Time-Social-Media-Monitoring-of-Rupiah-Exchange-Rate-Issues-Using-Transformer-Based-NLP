import pandas as pd
import regex as re


class TextPreprocessor:

    def __init__(self,df):

        self.df=df.copy()


    def select_columns(self):

        columns=[
            'id',
            'url',
            'text',
            'retweetCount',
            'replyCount',
            'likeCount',
            'quoteCount',
            'viewCount',
            'createdAt',
            'lang',
            'bookmarkCount'
        ]

        self.df=self.df[columns]

        return self


    def extract_author(self):

        self.df['author']=self.df['url'].str.extract(
            r"x\.com/([^/']+)"
        )

        self.df.drop(
            columns=['url'],
            inplace=True
        )

        return self


    def format_date(self):

        self.df['createdAt']=pd.to_datetime(
            self.df['createdAt']
        ).dt.date

        return self


    def filter_language(self):

        self.df=self.df.drop(
            self.df[
                self.df['lang']<'en'
            ].index
        )

        self.df=self.df[
            self.df['lang']=='in'
        ]

        self.df=self.df.reset_index(
            drop=True
        )

        self.df.drop(
            columns=['lang'],
            inplace=True
        )

        return self


    def remove_duplicate(self):

        self.df.drop_duplicates(
            inplace=True
        )

        return self


    def remove_null(self):

        self.df.dropna(
            inplace=True
        )

        return self


    def tweet_length(self):

        self.df["tweet_length"]=self.df[
            "text"
        ].astype(str).apply(len)

        return self


    def remove_outlier(self):

        self.df=self.df[
            (
                self.df["tweet_length"]>=15
            )
            &
            (
                self.df["tweet_length"]<=500
            )
        ]

        return self


    def sort_date(self):

        self.df=self.df.sort_values(
            by='createdAt'
        ).reset_index(
            drop=True
        )

        return self


    def get_data(self):

        return self.df