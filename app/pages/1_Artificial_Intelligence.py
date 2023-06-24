"""
Web application artificial intelligence page via streamlit.
"""


import os
from pathlib import Path
import sys


APP_PATH = Path(Path(os.path.realpath(__file__)).parents[1])
sys.path.append(APP_PATH)


from generate_page import generate_topic_page


generate_topic_page(topic="artificial_intelligence")