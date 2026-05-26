import pandas as pd
import regex as re
import emoji

from tqdm import tqdm
from transformers import pipeline


class SentimentAnalyzer:

    def __init__(self):

        self.classifier=pipeline(
            "sentiment-analysis",
            model="w11wo/indonesian-roberta-base-sentiment-classifier"
        )

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


    def preprocess(self,text):

        if pd.isna(text):
            return ""

        text=text.lower()
        text=re.sub(r"http\S+"," ",text)
        text=re.sub(r"@\w+"," ",text)
        text=re.sub(r"#"," ",text)
        text=re.sub(r"\brt\b"," ",text)

        text=emoji.replace_emoji(text,'')

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

        return text


    def normalize(self,text):

        for i in self.norm:
            text=text.replace(i,self.norm[i])

        return text


    def clean_text(self,df):

        df["clean_text"]=df[
            "text"
        ].apply(
            self.preprocess
        )

        df["clean_text"]=df[
            "clean_text"
        ].apply(
            self.normalize
        )

        return df


    def predict(self,df,batch_size=64):

        hasil_label=[]
        hasil_score=[]

        for i in tqdm(
            range(0,len(df),batch_size),
            desc="Proses Sentimen"
        ):

            teks_batch=(

                df["clean_text"]
                .iloc[i:i+batch_size]
                .fillna("")
                .astype(str)
                .tolist()

            )

            pred_batch=self.classifier(
                teks_batch,
                truncation=True,
                max_length=512
            )

            hasil_label.extend(
                [x["label"] for x in pred_batch]
            )

            hasil_score.extend(
                [x["score"] for x in pred_batch]
            )

            if len(hasil_label)%1000==0:

                temp_df=df.iloc[
                    :len(hasil_label)
                ].copy()

                temp_df[
                    "sentimen_bert"
                ]=hasil_label

                temp_df.to_csv(
                    "../data/processed/checkpoint_sentiment.csv",
                    index=False
                )

        df["sentimen_bert"]=hasil_label
        df["confidence"]=hasil_score

        return df