"""
Stubs for used tweepy functionality.
"""


import datetime
from typing import Optional, Protocol, TypedDict, Union


class _PublicMetrics(TypedDict):

    """
    Typed dict for tweet public metrics.
    """

    retweet_count: int
    reply_count: int
    like_count: int
    quote_count: int
    bookmark_count: int
    impression_count: int


class _TweetDataDict(TypedDict):

    """
    Typed dict for converted tweet to dict.
    """

    id: int
    text: str
    retweet_count: int
    quote_count: int
    reply_count: int
    like_count: int
    bookmark_count: int
    impression_count: int
    popularity: int
    created_at: datetime.datetime


class _TweepyReferencedTweetProtocol(Protocol):

    """
    Protocol for tweepy.ReferencedTweet object.
    """

    data: dict
    id: int
    type: Optional[str]


class _TweepyTweetProtocol(Protocol):

    """
    Protocol for tweepy tweet objects.
    """

    id: int
    text: str
    public_metrics: _PublicMetrics
    created_at: datetime.datetime
    referenced_tweets: list[_TweepyReferencedTweetProtocol]


class _TweepyClientProtocol(Protocol):
    
    """
    Protocol for tweepy.Client.
    """

    def search_recent_tweets(self) -> dict:
        ...


class _TweepyResponseProtocol(Protocol):

    """
    Protocol for tweet
    """

    data: list[_TweepyTweetProtocol]
    includes: dict
    errors: list
    meta: dict