"""
Script for running pipeline and updating data for app.
"""


import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional


from birdwatcher.config import PATHS
from birdwatcher.summarize import PostProcessor
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


APP_PATH = Path(os.path.join(
    Path(os.path.realpath(__file__)).parents[0],
    "app"
))
IMAGES_PATH = APP_PATH / "pages/images"


def _load_bool(value: str) -> bool:
    if value == "True":
        return True
    else:
        return False


def _load_latest_post_processor() -> PostProcessor:
    _logger.info(
        "Loading the latest PostProcessor object."
    )
    post_processor_path_list = list_s3_elements_with_pattern(
        path=PATHS.post_processing_path,
        pattern="PostProcessor_",
        file_extensions=[".pkl"]
    )["PostProcessor_"]
    post_processor_path = max(post_processor_path_list)
    return s3_load_pickle(key=post_processor_path)


def update_pie_charts(post_processor: Optional[PostProcessor] = None) -> None:
    """
    Update pie chart images displayed in topic pages.
    """
    if post_processor is None:
        post_processor = _load_latest_post_processor()
    
    _logger.info(
        "Updating pie chart images."
    )
    for key, value in tqdm(post_processor.pie_charts.items()):
        save_path = IMAGES_PATH / f"{key}" / f"{key}_pie_chart.png"
        value.fig.savefig(fname=save_path)
    return


def update_line_graphs(post_processor: Optional[PostProcessor] = None) -> None:
    """
    Update line graph images displayed in topic pages.
    """
    if post_processor is None:
        post_processor = _load_latest_post_processor()
    
    _logger.info(
        "Updating line graph images."
    )
    for key, value in tqdm(post_processor.line_graphs.items()):
        save_path = IMAGES_PATH / f"{key}" / f"{key}_line_graph.png"
        value.fig.savefig(fname=save_path)
    return


def update_plots(
    pie_charts: bool = True,
    line_graphs: bool = True,
    update_all: bool = True,
    post_processor: Optional[PostProcessor] = None
) -> None:
    """
    Update plot images displayed in topic pages.
    """
    if update_all or pie_charts:
        update_pie_charts(post_processor=post_processor)
    if update_all or line_graphs:
        update_line_graphs(post_processor=post_processor)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run twitter sentiment analysis pipeline."
    )
    parser.add_argument(
        "--update_plots", 
        type=_load_bool,
        help="Whether or not to update plot images for app.",
        required=False,
        default=True
    )
    args = parser.parse_args()

    # Update plots, if applicable.
    post_processor = _load_latest_post_processor()

    update_plots(
        update_all=args.update_plots,
        post_processor=post_processor
    )