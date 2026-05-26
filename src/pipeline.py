from pathlib import Path
import pandas as pd

from scraping.scraper import TwitterScraper
from preprocessing.text_processor import TextPreprocessor
from modeling.sentiment_model import SentimentAnalyzer
from modeling.topic_model import TopicAnalyzer
from modeling.crisis_detector import CrisisDetector


BASE_DIR=Path(__file__).resolve().parent.parent

RAW_PATH=(
    BASE_DIR/
    "data"/
    "raw"/
    "twitter_data.csv"
)

CLEAN_PATH=(
    BASE_DIR/
    "data"/
    "processed"/
    "twitter_data_cleaned.csv"
)

SENTIMENT_PATH=(
    BASE_DIR/
    "data"/
    "processed"/
    "sentiment_data.csv"
)

TOPIC_PATH=(
    BASE_DIR/
    "data"/
    "processed"/
    "topic_data.csv"
)

FINAL_PATH=(
    BASE_DIR/
    "data"/
    "processed"/
    "final_data.csv"
)

def main():

    print("="*50)
    print("STARTING PIPELINE")
    print("="*50)


    print("\n[1] Scraping Twitter Data...")

    scraper=TwitterScraper()

    df_raw=scraper.scrape(

        search_terms=[
            "rupiah",
            "dollar naik",
            "inflasi",
            "krisis ekonomi",
            "ekonomi Indonesia"
        ],

        start_date="2026-01-01",
        end_date="2026-05-22",

        max_items=2
    )

    scraper.save(
        df_raw,
        RAW_PATH
    )

    print(f"Total raw tweets: {len(df_raw)}")


    print("\n[2] Preprocessing...")

    df_clean=(

        TextPreprocessor(df_raw)

        .select_columns()
        .extract_author()
        .format_date()
        .filter_language()
        .remove_duplicate()
        .remove_null()
        .tweet_length()
        .remove_outlier()
        .sort_date()
        .get_data()

    )

    df_clean.to_csv(
        CLEAN_PATH,
        index=False
    )

    print(f"Total cleaned tweets: {len(df_clean)}")


    print("\n[3] Sentiment Analysis...")

    sentiment=SentimentAnalyzer()

    df_sentiment=sentiment.clean_text(
        df_clean
    )

    df_sentiment=sentiment.predict(
        df_sentiment
    )

    df_sentiment.to_csv(
        SENTIMENT_PATH,
        index=False
    )

    print("Sentiment completed")


    print("\n[4] Topic Modeling...")

    topic=TopicAnalyzer()

    df_topic=topic.clean_text(
        df_clean
    )

    df_topic=topic.remove_spam(
        df_topic
    )

    df_topic=topic.fit(
        df_topic
    )

    df_topic.to_csv(
        TOPIC_PATH,
        index=False
    )

    print("Topic modeling completed")


    print("\n[5] Crisis Detection...")

    crisis=CrisisDetector()

    df_final=crisis.merge_data(
        df_sentiment,
        df_topic
    )

    df_final=crisis.map_topic(
        df_final
    )

    df_final=crisis.topic_weight(
        df_final
    )

    df_final=crisis.sentiment_weight(
        df_final
    )

    df_final=crisis.calculate_crisis(
        df_final
    )

    daily_crisis=crisis.daily_crisis(
        df_final
    )

    df_final.to_csv(
        FINAL_PATH,
        index=False
    )

    print("Crisis detection completed")


    print("\n[6] Top Crisis Trigger")

    top_trigger=(

        df_final.groupby(
            "macro_topic"
        )["crisis_score"]

        .mean()

        .sort_values(
            ascending=False
        )

    )

    print(top_trigger)

    print("\nPipeline completed")
    print("="*50)


if __name__=="__main__":

    main()