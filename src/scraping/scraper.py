from apify_client import ApifyClient
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

class TwitterScraper:

    def __init__(self):

        self.client=ApifyClient(
            os.getenv("APIFY_TOKEN")
        )


    def scrape(
        self,
        search_terms,
        max_items=50
    ):

        run_input={

            "filter:blue_verified":False,
            "filter:consumer_video":False,
            "filter:has_engagement":False,
            "filter:hashtags":False,
            "filter:images":False,
            "filter:links":False,
            "filter:media":False,
            "filter:mentions":False,
            "filter:native_video":False,
            "filter:nativeretweets":False,
            "filter:news":False,
            "filter:pro_video":False,
            "filter:quote":False,
            "filter:replies":False,
            "filter:safe":False,
            "filter:spaces":False,
            "filter:twimg":False,
            "filter:videos":False,
            "filter:vine":False,
            "include:nativeretweets":False,

            "lang":"in",
            "maxItems":max_items,
            "queryType":"Latest",

            "searchTerms":search_terms

        }

        run=self.client.actor(
            "CJdippxWmn9uRfooo"
        ).call(
            run_input=run_input
        )

        items=[]

        for item in self.client.dataset(
            run.default_dataset_id
        ).iterate_items():

            items.append(item)

        return pd.DataFrame(items)


    def save(self,df,filepath):
        df.to_csv(
            filepath,
            index=False
        )
        print(
            f"Saved: {filepath}"
        )