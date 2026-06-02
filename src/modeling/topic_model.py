import pandas as pd
import numpy as np
import random
import re
import emoji

from bertopic import BERTopic
from umap import UMAP
from sentence_transformers import SentenceTransformer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class TopicAnalyzer:

    def __init__(self,seed=42):

        self.seed=seed

        np.random.seed(seed)
        random.seed(seed)

        self.norm={

            " gk ":" tidak ",
            " ga ":" tidak ",
            " gak ":" tidak ",
            " yg ":" yang ",
            " dpt ":" dapat ",
            " bgt ":" banget ",
            " krn ":" karena ",
            " udh ":" sudah ",
            " tdk ":" tidak ",
            " gt ":" gitu ",
            " klo ":" kalau ",
            " kalo ":" kalau ",
            " aja ":" saja ",
            " nih ":" ini ",
            " ama ":" sama ",
            " ma ":" sama ",
            " siaapp ":" siap ",
            " bnr ":" benar ",
            " gpp ":" tidak apa-apa ",
            " bener ":" benar ",
            " dr ":" dari ",
            " hrs ":" harus ",
            " nnti ":" nanti ",
            " nnt ":" nanti ",
            " dgn ":" dengan ",
            " sm ":" sama ",
            " sy ":" saya ",
            " lg ":" lagi ",
            " hrsnya ":" seharusnya ",
            " hrsnha ":" seharusnya ",
            " hrsnyaa ":" seharusnya ",
            " hrsnyaaa ":" seharusnya ",
            " sblm ":" sebelum ",
            " sblmnya ":" sebelumnya ",
            " bgttt ":" banget ",
            " tpi ":" tapi ",
            " tp ":" tapi ",
            " bgtu ":" begitu ",
            " blm ":" belum ",
            " gmn ":" gimana ",
            " mls ":" malas ",
            "tp ":" tapi ",
            "tpi ":" tapi ",
            " kpd ":" kepada ",
            " brp ":" berapa ",
            " bs ":" bisa ",
            " mo ":" mau ",
            " krj ":" kerja "

        }

        factory=StopWordRemoverFactory()

        self.stop_words=set(factory.get_stop_words())

        more_stop_words={

            "rt",
            "amp",
            "gue",
            "gw",
            "gua",
            "wkwk",
            "wk",
            "wkwkwk",
            "haha",
            "hehe",
            "lol",
            "bro",
            "min",
            "nya"

        }

        self.stop_words.update(more_stop_words)

        self.embedding_model=SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=seed
        )

        self.topic_model=BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            language="multilingual",
            min_topic_size=15,
            nr_topics=None,
            calculate_probabilities=True,
            verbose=True
        )


    def preprocess(self,text):

        if pd.isna(text):
            return ""

        text=text.lower()
        text=re.sub(r"http\S+"," ",text)
        text=re.sub(r"@\w+"," ",text)
        text=re.sub(r"#"," ",text)
        text=re.sub(r"\brt\b"," ",text)

        text=emoji.replace_emoji(text,"")

        text=re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            text
        )

        text=re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        text=re.sub(
            r"\b\d+\b",
            "",
            text
        )

        return text


    def normalize(self,text):

        for i in self.norm:
            text=text.replace(i,self.norm[i])

        return text


    def remove_stopwords(self,text):

        words=text.split()

        filtered_words=[
            word.lower()
            for word in words
            if word.lower() not in self.stop_words
        ]

        return " ".join(filtered_words)


    def clean_text(self,df):

        df["clean_text"]=df["text"].apply(self.preprocess)
        df["clean_text"]=df["clean_text"].apply(self.normalize)

        df["clean_text"]=(
            df["clean_text"]
            .astype(str)
            .apply(self.remove_stopwords)
        )

        df=df[
            df["clean_text"]
            .str.strip()!=""
        ]

        df.drop_duplicates(
            subset=["clean_text"],
            inplace=True
        )

        df.dropna(inplace=True)

        return df


    def remove_spam(self,df):

        spam_words=[
            "download",
            "saldo",
            "promo",
            "shopeepay",
            "voucher",
            "zimbabwe",
            "world",
            "largest"
        ]

        pattern="|".join(spam_words)

        df=df[
            ~df["clean_text"]
            .str.contains(
                pattern,
                case=False,
                na=False
            )
        ]

        return df.reset_index(drop=True)


    def fit(self,df):

        documents=(
            df["clean_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        topics,probs=self.topic_model.fit_transform(
            documents
        )

        df["topic"]=topics

        return df


    def load(self,path):

        self.topic_model=BERTopic.load(
            str(path),
            embedding_model=self.embedding_model
        )


    def save(self,path):

        self.topic_model.save(
            str(path),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=False
        )


    def transform(self,df):

        documents=(
            df["clean_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        topics,_=self.topic_model.transform(documents)

        df["topic"]=topics

        return df


    def get_model(self):

        return self.topic_model