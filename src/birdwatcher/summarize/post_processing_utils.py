"""
Utilities for post-processing.
"""


from dataclasses import dataclass, field
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd
import pickle
from typing import Optional


from birdwatcher.config import PATHS
from birdwatcher.ml.inference import _get_scores_save_path
from birdwatcher.s3_utils import (
    list_s3_elements_with_pattern,
    s3_load_pickle
)


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


with open(PATHS.dc_run_date_info, "rb") as infile:
    run_info = pickle.load(infile)


@dataclass(frozen=True)
class SentimentDistribution:

    num_negative: int
    num_positive: int


@dataclass(frozen=True)
class SentimentLineGraph:

    fig: matplotlib.figure.Figure
    positive_list: list[float]
    date_list: list[str]


@dataclass(frozen=True)
class SentimentPieChart:
    
    fig: matplotlib.figure.Figure
    ax: plt.Axes
    patches: list
    texts: list
    autotexts: list


def _generate_sentiment_pie_chart(
    data: npt.ArrayLike,
    data_key: str,
    end_date_name: str,
    params: dict
) -> SentimentPieChart:
    """
    Generate a sentiment pie chart object for a given data key.
    """
    fig, ax = plt.subplots()
    patches, texts, autotexts = ax.pie(
        x=data,
        autopct=params["autopct"],
        colors=params["colors"],
        explode=params["explode"],
        shadow=params["shadow"],
        wedgeprops=params["wedge_properties"]
    )
    ax.legend(
        patches,
        params["labels"],
        fontsize=params["legend_fontsize"],
        title=params["legend_title"],
        loc=params["loc"]
    )
    plt.setp(autotexts, weight="bold")
    ax.set_title(
        params["title"].format(
            data_key=data_key.replace("_", " ").title()
        )
    )
    return SentimentPieChart(
        fig=fig,
        ax=ax,
        patches=patches,
        texts=texts,
        autotexts=autotexts
    )


def _generate_sentiment_distribution(
    data_key: str,
    end_date_name: str
) -> SentimentDistribution:
    """
    Generate a sentiment distribution object for a given data key.
    """
    scores_path = _get_scores_save_path(
        data_key=data_key,
        end_date_name=end_date_name
    )
    df_scores = pd.read_parquet(scores_path)
    num_positive = df_scores.preds.sum()
    num_negative = len(df_scores.index) - num_positive
    return SentimentDistribution(
        num_negative=num_negative,
        num_positive=num_positive
    )


def _load_recent_postprocessors(
    max_return: Optional[int] = None
) -> list:
    """
    Load recent PostProcessor objects.
    """
    post_processor_path_list = list_s3_elements_with_pattern(
        path=PATHS.post_processing_path,
        pattern="PostProcessor_",
        file_extensions=[".pkl"]
    )["PostProcessor_"]
    post_processor_path_list.sort(reverse=True)
    
    if max_return:
        post_processor_list = [
            s3_load_pickle(key=pp)
            for idx, pp in enumerate(post_processor_path_list)
            if idx < max_return
        ]
    else:
        post_processor_list = [
            s3_load_pickle(key=pp)
            for pp in post_processor_path_list
        ]
    return post_processor_list


def _generate_history_line_graph(
    data_key: str,
    params: dict
) -> SentimentLineGraph:
    """
    Generate a historical line graph of positive sentiment.
    """
    post_processor_list = _load_recent_postprocessors(
        max_return=params["max_records"]
    )
    positive_list = [
        (pp.distributions[data_key].num_positive / (
            pp.distributions[data_key].num_negative +
            pp.distributions[data_key].num_positive
        ))
        for pp in post_processor_list 
    ]
    date_list = [
        pp.end_date_name for pp in post_processor_list
    ]
    title = (
        params["title"].format(data_key=data_key).replace("_", " ").title()
    )

    fig = plt.figure()
    plt.plot(date_list, positive_list, marker=params["marker"])
    plt.ylim(params["ylim"])
    plt.xlabel(params["xlabel"])
    plt.ylabel(params["ylabel"])
    plt.title(title)
    return SentimentLineGraph(
        fig=fig,
        positive_list=positive_list,
        date_list=date_list
    )