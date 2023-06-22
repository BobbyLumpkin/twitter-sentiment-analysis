"""
Generate summaries of scores.
"""


from dataclasses import dataclass, field
import logging
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from typing import Optional, Union


from birdwatcher.config import PATHS, SUMMARIZE_CONFIG
from birdwatcher.ml.inference import _get_scores_save_path
from birdwatcher.s3_utils import s3_save_pickle, s3_load_pickle
from birdwatcher.summarize.post_processing_utils import (
    _generate_history_line_graph,
    _generate_sentiment_distribution,
    _generate_sentiment_pie_chart,
    SentimentDistribution,
    SentimentLineGraph,
    SentimentPieChart
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


@dataclass
class PostProcessor:

    """
    A post-processing object.
    """

    data_key_list: list[str] = field(
        default_factory=lambda: SUMMARIZE_CONFIG.data_key_list
    )
    distributions: dict[str, SentimentDistribution] = field(
        default_factory=lambda: {}
    )
    end_date_name: str = field(default=run_info.end_date_name)
    line_graphs: dict[str, SentimentLineGraph] = field(default_factory=lambda: {})
    line_graph_config: dict = field(
        default_factory=lambda: SUMMARIZE_CONFIG.line_graph_params
    )
    pie_charts: dict[str, SentimentPieChart] = field(default_factory=lambda: {})
    pie_chart_config: dict = field(
        default_factory=lambda: SUMMARIZE_CONFIG.pie_chart_params
    )

    def __repr__(self):
        return (
            f"PostProcessor(data_key_list={self.data_key_list}, "
            f"end_date_name='{self.end_date_name}')"
        )

    def _update_distribution(
        self,
        data_key: str,
        end_date_name: Optional[str] = None,
        verbose: bool = False
    ) -> None:
        """
        Update distribution attribute for a given data key.
        """
        # Define defaults using attributes of self.
        if end_date_name is None:
            end_date_name = self.end_date_name
        
        # Update distribution.
        if verbose:
            _logger.info(
                f"Updating score distribution for data_key: {data_key}."
            )
        self.distributions[data_key] = _generate_sentiment_distribution(
            data_key=data_key,
            end_date_name=end_date_name
        )
        return

    def update_distributions(
        self,
        data_key_list: Optional[list[str]] = None,
        end_date_name: Optional[str] = None,
        verbose: bool = False
    ) -> None:
        """
        Update distributions attribute for a list of data keys.
        """
        # Define defaults using attributes of self.
        if data_key_list is None:
            data_key_list = self.data_key_list
        if end_date_name is None:
            end_date_name = self.end_date_name
        
        # Update distributions.
        for data_key in data_key_list:
            self._update_distribution(
                data_key=data_key,
                end_date_name=end_date_name,
                verbose=verbose
            )
        return
    
    def _update_line_graph(
        self,
        data_key: str,
        end_date_name: Optional[str] = None,
        line_graph_params: Optional[dict] = None,
        verbose: bool = False
    ) -> None:
        """
        Generate a line graph, visualizing historical trends.
        """
        # Define defaults using attributes of self.
        if end_date_name is None:
            end_date_name = self.end_date_name
        if line_graph_params is None:
            line_graph_params = self.line_graph_config
        
        # Update line graph.
        if verbose:
            _logger.info(
                f"Updating line graph for data_key: {data_key}."
            )
        self.line_graphs[data_key] = _generate_history_line_graph(
            data_key=data_key,
            params=line_graph_params
        )
        return

    def update_line_graphs(
        self,
        data_key_list: Optional[list[str]] = None,
        end_date_name: Optional[str] = None,
        line_graph_params: Optional[dict] = None,
        verbose: bool = False
    ) -> None:
        """
        Update line_graphs attribute for a list of data keys.
        """
        # Define defaults using attributes of self.
        if data_key_list is None:
            data_key_list = self.data_key_list
        if end_date_name is None:
            end_date_name = self.end_date_name
        if line_graph_params is None:
            line_graph_params = self.line_graph_config
        
        # Update pie charts.
        for data_key in data_key_list:
            self._update_line_graph(
                data_key=data_key,
                end_date_name=end_date_name,
                line_graph_params=line_graph_params,
                verbose=verbose
            )
        return

    def _update_pie_chart(
        self,
        data_key: str,
        end_date_name: Optional[str] = None,
        pie_chart_params: Optional[dict] = None,
        verbose: bool = False
    ) -> None:
        """
        Generate a pie chart, visualizing sentiment distribution.
        """
        # Define defaults using attributes of self.
        if end_date_name is None:
            end_date_name = self.end_date_name
        if pie_chart_params is None:
            pie_chart_params = self.pie_chart_config
        
        # Update pie chart.
        data = [self.distributions[data_key].num_positive,
                self.distributions[data_key].num_negative]
        if verbose:
            _logger.info(
                f"Updating pie chart for data_key: {data_key}."
            )
        self.pie_charts[data_key] = _generate_sentiment_pie_chart(
            data=data,
            data_key=data_key,
            end_date_name=end_date_name,
            params=pie_chart_params
        )
        return
    
    def update_pie_charts(
        self,
        data_key_list: Optional[list[str]] = None,
        end_date_name: Optional[str] = None,
        pie_chart_params: Optional[dict] = None,
        verbose: bool = False
    ) -> None:
        """
        Generate pie charts attribute for a list of data keys.
        """
        # Define defaults using attributes of self.
        if data_key_list is None:
            data_key_list = self.data_key_list
        if end_date_name is None:
            end_date_name = self.end_date_name
        if pie_chart_params is None:
            pie_chart_params = self.pie_chart_config
        
        # Update pie charts.
        for data_key in data_key_list:
            self._update_pie_chart(
                data_key=data_key,
                end_date_name=end_date_name,
                pie_chart_params=pie_chart_params,
                verbose=verbose
            )
        return
    
    def save(
        self,
        bucket: str = PATHS.s3_bucket.replace("s3://", ""),
        key: str = PATHS.post_processing_path,
        file_name: str = None,
        save_path: str = None
    ) -> None:
        """
        Save self as pickle object.
        """
        if save_path is None:
            if file_name is None:
                file_name = f"PostProcessor_{self.end_date_name}.pkl"
            _logger.info(
                f"Saving to bucket: {bucket} with key: "
                f"{PATHS.post_processing_path} with filename: "
                f"{file_name}"
            )
            s3_save_pickle(
                obj=self,
                bucket=bucket,
                key=PATHS.post_processing_path,
                file_name=file_name
            )
        else:
            _logger.info(
                f"Saving to {save_path}."
            )
            with open(save_path, "wb") as outfile:
                pickle.dump(self, outfile)
        return
