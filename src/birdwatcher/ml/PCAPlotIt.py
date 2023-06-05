"""
Purpose: Subclass of sklearn.decomposition.PCA for quick plotting.
"""


from dataclasses import dataclass, field
import logging
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


from birdwatcher.config import PATHS


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s:%(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


@dataclass
class PCAPlotIt(PCA):

    """
    Subclass of PCA for quick plotting.
    """

    def __init__(self, cumulative_variance_target: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.cumulative_variance_target = cumulative_variance_target
        self.fit_status = False
        return

    def fit(self, X, y=None):
        super().fit(X, y)
        self.fit_status = True
        self.train_shape_0 = X.shape[0]
        self.train_shape_1 = X.shape[1]
        self.pc_id = np.arange(min(self.train_shape_0, self.train_shape_1)) + 1
        self.cum_var = np.cumsum(self.explained_variance_ratio_)
        self.num_pcs_retained = np.where(
            self.cum_var > self.cumulative_variance_target
        )[0][0] + 1

        _logger.info(
            f"Target cumulative variance: {self.cumulative_variance_target}"
        )
        _logger.info(
            f"Number of principal components: {self.num_pcs_retained}"
        )
        return

    def _generate_scree_plot(self):
        # Generate scree plot.
        self.fig = plt.figure(figsize=(14,6))
        ax1 = self.fig.add_subplot(121)
        ax1.plot(self.pc_id, self.explained_variance_ratio_, 'ro-', linewidth=2)
        ax1.set_xlabel("Principal components")
        ax1.set_ylabel("Variance")
        ax1.set_title("Scree Plot", size=20)
        return
    
    def _generate_cumulative_variance_plot(self):
        # Generate cumulative variance plot.
        ax2 = self.fig.add_subplot(122)
        ax2.plot(self.pc_id, self.cum_var, 'ro-', linewidth=2)
        ax2.set_xlabel("Principal components")
        ax2.set_ylabel("Cumulative variance")
        ax2.set_xlim()
        ax2.set_ylim()
        ax2.hlines(0.8, ax2.get_xlim()[0], ax2.get_xlim()[1], linestyles = "--", colors = plt.cm.Greys(200))
        ax2.vlines(self.num_pcs_retained, ax2.get_ylim()[0], ax2.get_ylim()[1], linestyles = "--", colors = plt.cm.Greys(200))
        ax2.plot(self.num_pcs_retained, 0.8, color = 'dimgrey', marker = '*', markersize = 30)
        ax2.set_title("Cumulative Variance Explained", size = 20)
        return 

    def generate_plots(self):
        """
        Generate scree and cumulative variance plots.
        """
        if not self.fit_status:
            raise AttributeError("Must fit the PCA_plus object before generating plots.")
            
        self._generate_scree_plot()
        self._generate_cumulative_variance_plot()
        return
    
    def transform(self, X):
        """
        Compute the first self.num_pcs_retained PCs.
        """
        pcs = super().transform(X)
        pc_list = [
            f"PC-{comp}"
            for comp in np.arange(1, self.num_pcs_retained + 1)
        ]
        df_retained_pcs = pd.DataFrame(
            pcs[:, 0 : self.num_pcs_retained],
            columns=pc_list
        )
        return df_retained_pcs