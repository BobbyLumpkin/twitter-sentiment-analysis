"""
Generate summaries of scores.
"""


import logging
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from typing import Union


from birdwatcher.config import PATHS
from birdwatcher.ml.inference import _get_scores_save_path



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


def generate_save_pie_chart(
    data_key: str,
    end_date_name: str = run_info.end_date_name,
    save: bool = True
):
    """
    Generate a pie chart, visualizing sentiment distribution.
    """
    scores_path = _get_scores_save_path(
        data_key=data_key,
        end_date_name=end_date_name
    )
    df_scores = pd.read_parquet(scores_path)
    num_positive = df_scores.preds.sum()
    num_negative = len(df_scores.index) - num_positive
    labels = ["Positive", "Negative"]
    fig, ax = plt.subplots()
    ax.pie(
        [num_positive, num_negative],
        labels=labels,
        autopct='%1.1f%%',
        colors=["green", "red"]
    )
    ax.set_title(f"{data_key.capitalize()} Sentiment Distribution", size=20)
    return 
