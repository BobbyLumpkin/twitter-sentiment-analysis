"""
Call twitter API and convert results into pandas dataframes.
"""


from dataclasses import dataclass, field
from datetime import datetime
import logging
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import tweepy
from typing import Optional, Union


from birdwatcher.config import DATA_COLLECTION_CONFIG, PATHS, SECRETS
from birdwatcher.datacollection.type_utils import (
    _PublicMetrics,
    _TweetDataDict,
    _TweepyTweetProtocol,
    _TweepyClientProtocol,
    _TweepyResponseProtocol
)


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


@dataclass
class CollectionRunInfo():

    """
    Store of data collection run information.
    """

    start_date: Optional[str]
    end_date: str
    save_path: Union[str, Path] = field(default=PATHS.dc_run_date_info)

    def __post_init__(self):
        self._generate_end_date_name()
        return
    
    def _generate_end_date_name(self) -> None:
        """
        Generate end_date_name from end_date.
        """
        end_date_dt = datetime.strptime(
            self.end_date, "%Y-%m-%dT%H:%M:%SZ"
        )
        setattr(self, "end_date_name", end_date_dt.strftime("%Y%m%d"))
        return

    def save(
        self,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Save self as pickle file.
        """
        if path is None:
            path = self.save_path
        with open(path, "wb") as outfile: 
            pickle.dump(self, outfile)
        _logger.info(
            f"Saved CollectionRunInfo object to {path!r}."
        )
        return


def tweet_data_to_dict(
    tweet_data: _TweepyTweetProtocol
) -> _TweetDataDict:
    """
    Convert element from paginator into dictionary.
    """
    return {
        "id": tweet_data.id,
        "text": tweet_data.text,
        "retweet_count": tweet_data.public_metrics["retweet_count"],
        "quote_count": tweet_data.public_metrics["quote_count"],
        "reply_count": tweet_data.public_metrics["reply_count"],
        "like_count": tweet_data.public_metrics["like_count"],
        "bookmark_count": tweet_data.public_metrics["bookmark_count"],
        "impression_count": tweet_data.public_metrics["impression_count"],
        "popularity": sum(tweet_data.public_metrics.values()),
        "created_at": tweet_data.created_at
    }


def convert_tweets_to_df(
    client: _TweepyClientProtocol, 
    query: str, 
    max_results: int, 
    limit: int, 
    start_date: Union[datetime, str], 
    end_date: Union[datetime, str]
):
    """
    Get tweet data and format as pandas dataframe.
    """
    tweets_dict_list = [
        tweet_data_to_dict(tweet) 
        for tweet in tqdm(list(tweepy.Paginator(
            client.search_recent_tweets, 
            query=query,
            start_time=start_date,
            end_time=end_date,
            tweet_fields=[
                "public_metrics",
                "id",
                "text",
                "created_at",
                "referenced_tweets"
            ], 
            max_results=max_results
        ).flatten(limit=limit)), desc=f"Retreiving results for '{query}'")
        if tweet.referenced_tweets is None
    ]
    tweets_df = pd.DataFrame(tweets_dict_list)
    return tweets_df


def request_save_queries(
    key_list: list = DATA_COLLECTION_CONFIG.query_info.keys(),
    query_info_dict: dict = DATA_COLLECTION_CONFIG.query_info,
    start_date: Union[datetime, str] = DATA_COLLECTION_CONFIG.start_date,
    end_date: Union[datetime, str] = DATA_COLLECTION_CONFIG.end_date,
    max_results: int = 100, 
    limit: int = 10000, 
    save: bool = True,
    run_info_save_path: str = None,
    client: _TweepyClientProtocol = None
) -> CollectionRunInfo:
    """
    Make twitter API requests, convert to pandas dataframes and save.
    
    Args:
        key_list: List of keys of 'query_info' dict to
            retrieve tweets for.
        query_info: A dictionary of info for each data-key in
            'key_list'.
        start_date: The oldest UTC timestamp (from most
            recent seven days) from which the Tweets will be provided. (see
            the documentation for tweepy.client.search_recent_tweets for
            more information.)
        end_date: The newest, most recent UTC timestamp to
            which the Tweets will be provided. (see the documentation for
            tweepy.client.search_recent_tweets for more information.)
        max_results: The maximum number of search results to be
            returned by a request. A number between 10 and 100. (defaults
            to 100.) (see the documentation for
            tweepy.client.search_recent_tweets for more information.)
        limit: The maximum number of results to yield. (defaults to
            10000) (see the documentation for tweepy.Paginator for more
            information.)
        save: Whether or not to save query results.
        run_info_save_path: Path to save 'data_collection_run_info'
            object to.
        client: A tweepy.Client object.
    
    Returns:
        run_info: An instance of
            `birdwatcher.datacollection.CollectionRunInfo`.
    """
    # Get tweepy client, if not provided.
    if client is None:
        client  = tweepy.Client(
            bearer_token=SECRETS.bearer_token,
            wait_on_rate_limit=True
        )
    # Convert start/end dates to datetimes & instantiate
    # CollectionRunInfo object.
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
    run_info = CollectionRunInfo(
        start_date=start_date,
        end_date=end_date
    )

    # Loop through queries in query_list.
    for key in key_list:
        _logger.info(f"Acquiring tweets for key: {key}.")
        query_info = query_info_dict[key]
        query_list = query_info["query_list"]
        df_list = []
        # Retrieve df for each query in query_list
        for query in query_list:
            df = convert_tweets_to_df(
                client, 
                query, 
                max_results=max_results, 
                limit=limit, 
                start_date=start_date_dt, 
                end_date=end_date_dt
            )
            df_list.append(df)
        df_query = pd.concat(df_list, axis=0, ignore_index=True)
        df_query.drop_duplicates(subset="id", inplace=True)
        _logger.info(f"Queries returned {len(df_query.index)} rows.")

        # Save df_query, if applicable.
        if save: 
            fpath_raw = query_info["fpath_raw"]
            file_prefix = query_info["file_prefix"]
            fname = f"{file_prefix}_{run_info.end_date_name}_raw.parquet"
            save_path = f"{fpath_raw}/{fname}"
            df_query.to_parquet(save_path)
            _logger.info(f"{key} tweets saved to storage: {save_path}")
    
    # Save CollectionRunInfo object, if applicable.
    if save:
        run_info.save(path=run_info_save_path)
    return

