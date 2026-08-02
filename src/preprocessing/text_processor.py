import pandas as pd
import regex as re


class TextPreprocessor:

    def __init__(self,df):

        self.df=df.copy()


    def select_columns(self):

        columns = [
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

        aliases = {
            "url": ["twitterUrl"],
            "createdAt": ["created_at", "date"],
            "lang": ["language", "langCode"],
        }

        defaults = {
            "retweetCount": 0,
            "replyCount": 0,
            "likeCount": 0,
            "quoteCount": 0,
            "viewCount": 0,
            "bookmarkCount": 0,
            "createdAt": pd.Timestamp.utcnow().isoformat(),
            "lang": "in",
        }

        selected = {}
        for column in columns:
            if column in self.df.columns:
                selected[column] = self.df[column]
                continue

            alias = next(
                (alt for alt in aliases.get(column, []) if alt in self.df.columns),
                None
            )
            if alias is not None:
                selected[column] = self.df[alias]
                continue

            selected[column] = defaults.get(column, pd.NA)

        self.df = pd.DataFrame(selected)

        return self


    def extract_author(self):

        self.df['author'] = self.df['url'].fillna("").astype(str).str.extract(
            r"(?:x|twitter)\.com/([^/']+)"
        )

        self.df['author'] = self.df['author'].replace("", pd.NA).fillna("unknown")

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