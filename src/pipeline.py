import json
from pathlib import Path
from datetime import date

import pandas as pd

from scraping.scraper import TwitterScraper
from preprocessing.text_processor import TextPreprocessor
from modeling.sentiment_model import SentimentAnalyzer
from modeling.topic_model import TopicAnalyzer
from modeling.crisis_detector import CrisisDetector


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH       = BASE_DIR / "data" / "raw"       / "twitter_data.csv"
CLEAN_PATH     = BASE_DIR / "data" / "processed" / "twitter_data_cleaned.csv"
SENTIMENT_PATH = BASE_DIR / "data" / "processed" / "sentiment_data.csv"
TOPIC_PATH     = BASE_DIR / "data" / "processed" / "topic_data.csv"
FINAL_PATH     = BASE_DIR / "data" / "processed" / "final_data.csv"
TOPIC_MODEL_PATH = BASE_DIR / "data" / "topic_model"
STATE_PATH     = BASE_DIR / "data" / "state.json"


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_run": None, "topic_model_fitted": False}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def append_deduplicate(new_df, existing_path):
    """Append new_df to existing CSV, deduplicate by 'id', return merged df."""
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged.drop_duplicates(subset=["id"], keep="last", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged
    return new_df.copy()


def filter_new_only(df, existing_path):
    """Return only rows whose 'id' is not already in existing_path."""
    if not existing_path.exists():
        return df
    existing_ids = set(pd.read_csv(existing_path, usecols=["id"])["id"])
    return df[~df["id"].isin(existing_ids)].reset_index(drop=True)


def main():

    print("=" * 50)
    print("STARTING PIPELINE")
    print("=" * 50)

    state = load_state()
    is_first_run = not state["topic_model_fitted"]

    # ------------------------------------------------------------------
    # [1] Scrape
    # ------------------------------------------------------------------
    print("\n[1] Scraping Twitter Data...")

    scraper = TwitterScraper()

    df_raw_new = scraper.scrape(
        search_terms=[
            "rupiah",
            "dollar naik",
            "inflasi",
            "krisis ekonomi",
            "ekonomi Indonesia"
        ],
        max_items=50
    )

    print(f"New raw tweets fetched: {len(df_raw_new)}")

    # Filter to truly new tweets before any processing
    df_raw_new = filter_new_only(df_raw_new, RAW_PATH)
    print(f"New tweets after dedup against existing raw: {len(df_raw_new)}")

    df_raw = append_deduplicate(df_raw_new, RAW_PATH)
    df_raw.to_csv(RAW_PATH, index=False)

    print(f"Total raw tweets (cumulative): {len(df_raw)}")

    # ------------------------------------------------------------------
    # [2] Preprocess — only new tweets
    # ------------------------------------------------------------------
    print("\n[2] Preprocessing...")

    df_clean_new = (
        TextPreprocessor(df_raw_new)
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

    # Drop any that somehow already exist in the cleaned CSV
    df_clean_new = filter_new_only(df_clean_new, CLEAN_PATH)

    df_clean = append_deduplicate(df_clean_new, CLEAN_PATH)
    df_clean.to_csv(CLEAN_PATH, index=False)

    print(f"New cleaned tweets: {len(df_clean_new)}")
    print(f"Total cleaned tweets (cumulative): {len(df_clean)}")

    if df_clean_new.empty:
        print("No new tweets to process. Exiting.")
        return

    # ------------------------------------------------------------------
    # [3] Sentiment — only new tweets
    # ------------------------------------------------------------------
    print("\n[3] Sentiment Analysis...")

    sentiment = SentimentAnalyzer()

    df_sentiment_new = sentiment.clean_text(df_clean_new.copy())
    df_sentiment_new = filter_new_only(df_sentiment_new, SENTIMENT_PATH)
    df_sentiment_new = sentiment.predict(df_sentiment_new)

    df_sentiment = append_deduplicate(df_sentiment_new, SENTIMENT_PATH)
    df_sentiment.to_csv(SENTIMENT_PATH, index=False)

    print("Sentiment completed")

    # ------------------------------------------------------------------
    # [4] Topic Modeling — fit once, transform on subsequent runs
    # ------------------------------------------------------------------
    print("\n[4] Topic Modeling...")

    topic = TopicAnalyzer()

    df_topic_new = topic.clean_text(df_clean_new.copy())
    df_topic_new = topic.remove_spam(df_topic_new)
    df_topic_new = filter_new_only(df_topic_new, TOPIC_PATH)

    if is_first_run:
        print("First run — fitting topic model on all cleaned data...")

        # Fit on all historical data so the model is representative
        df_all_for_fit = topic.clean_text(df_clean.copy())
        df_all_for_fit = topic.remove_spam(df_all_for_fit)
        df_all_for_fit = topic.fit(df_all_for_fit)

        topic.save(TOPIC_MODEL_PATH)

        # Derive topic assignments for the new batch from the fitted model
        df_topic_new = topic.transform(df_topic_new)

        state["topic_model_fitted"] = True

    else:
        print("Subsequent run — transforming with saved topic model...")
        topic.load(TOPIC_MODEL_PATH)
        df_topic_new = topic.transform(df_topic_new)

    df_topic = append_deduplicate(df_topic_new, TOPIC_PATH)
    df_topic.to_csv(TOPIC_PATH, index=False)

    print("Topic modeling completed")

    # ------------------------------------------------------------------
    # [5] Crisis Detection — only new tweets, append to final
    # ------------------------------------------------------------------
    print("\n[5] Crisis Detection...")

    crisis = CrisisDetector()

    df_final_new = crisis.merge_data(df_sentiment_new, df_topic_new)
    df_final_new = filter_new_only(df_final_new, FINAL_PATH)
    df_final_new = crisis.map_topic(df_final_new)
    df_final_new = crisis.topic_weight(df_final_new)
    df_final_new = crisis.sentiment_weight(df_final_new)
    df_final_new = crisis.calculate_crisis(df_final_new)

    df_final = append_deduplicate(df_final_new, FINAL_PATH)

    # Re-apply macro_topic mapping to the full dataset on every run
    # so the mapping stays consistent even if crisis_detector.py changes
    df_final = crisis.map_topic(df_final)
    df_final = crisis.topic_weight(df_final)
    df_final = crisis.sentiment_weight(df_final)
    df_final = crisis.calculate_crisis(df_final)

    df_final.to_csv(FINAL_PATH, index=False)

    print("Crisis detection completed")

    # ------------------------------------------------------------------
    # [6] Summary
    # ------------------------------------------------------------------
    print("\n[6] Top Crisis Trigger")

    top_trigger = (
        df_final
        .groupby("macro_topic")["crisis_score"]
        .mean()
        .sort_values(ascending=False)
    )

    print(top_trigger)

    state["last_run"] = date.today().isoformat()
    save_state(state)

    print(f"\nPipeline completed — last_run saved as {state['last_run']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
