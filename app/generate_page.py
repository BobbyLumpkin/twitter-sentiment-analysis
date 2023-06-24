"""
Utilities for generating pages.
"""


from PIL import Image
import streamlit as st


from birdwatcher.config import DATA_COLLECTION_CONFIG, PATHS
from birdwatcher.s3_utils import (
    list_s3_elements_with_pattern,
    s3_load_pickle
)


PAGE_INTRO_MARKDOWN = """
The plots below visualize the distribution of positive and negative
sentiment tweets referring to {topic} and how they've changed over
time. Tweets were flagged as pertaining to {topic} if they included
any of the following keywords: 
"""

PIE_CHART_CAPTION = """
Distribution of {topic} tweet sentiments for the latest week. 
"""

LINE_GRAPH_CAPTION = """
Historical trend of sentiment distributions for {topic} tweets.
"""


def _generate_page_intro_markdown(topic: str) -> str:
    """
    Generate a markdown list of keywords used to identify topic tweets.
    """
    TOPIC_TEXT = topic.replace("_", " ")
    keyword_list = DATA_COLLECTION_CONFIG.query_info[topic]["query_list"]
    markdown = PAGE_INTRO_MARKDOWN.format(topic=TOPIC_TEXT)
    for keyword in keyword_list:
        markdown += f"\n* '{keyword}'"
    return markdown


def generate_topic_page(topic: str) -> None:
    """
    Generate a topic page.
    """
    TOPIC_TEXT = topic.replace("_", " ")
    # Set the main page config.
    st.set_page_config(
        page_title="Let's go Birdwatching",
        page_icon="../images/icon.png"
    )

    # Create a title.
    st.title(topic.replace("_", " ").title())
    st.sidebar.success("Select a topic above.")

    # Display the pie chart.
    intro_markdown = _generate_page_intro_markdown(topic=topic)
    st.write(intro_markdown)
    pie_chart = Image.open(f"./images/{topic}/{topic}_pie_chart.png")
    st.image(pie_chart, caption=PIE_CHART_CAPTION.format(topic=TOPIC_TEXT))
    
    # Display the line graph.
    line_graph = Image.open(f"./images/{topic}/{topic}_line_graph.png")
    st.image(line_graph, caption=LINE_GRAPH_CAPTION.format(topic=TOPIC_TEXT))
    return