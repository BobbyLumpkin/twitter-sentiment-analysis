"""
Web application home page via streamlit.
"""


from PIL import Image
import streamlit as st


HOME_PAGE_MARKDOWN = """
# Birdwatcher

## Welcome!

Welcome my fellow bird enthusiasts 🐦😂! All joking aside, this is a
web app designed to display sentiment distributions associated with
particular topics on twitter. (i.e. birdwatching; get it 😉?) 

## Navigating The App

Over on the left-hand side, you'll find a list of topics I chose to
monitor. Clicking on one will take you to page displaying sentiment
distribution visualizations. You'll find two plots:

1. A pie chart, indicating the sentiment distribution using recent
(within the last week) tweets only.
2. A line graph, visualizing changes in this distribution over time.

## Links

The source code for both the sentiment analysis pipeline and app
front-end can be found in my GitHub: [twitter-sentiment-analysis](https://github.com/BobbyLumpkin/twitter-sentiment-analysis/tree/main)
"""

# Set the main page config.
st.set_page_config(
    page_title="Let's go Birdwatching",
    page_icon="./images/icon.png"
)

# Write intro page markdown.
col1, col2 = st.columns([7, 3])

with col1:
    st.write(HOME_PAGE_MARKDOWN)

with col2:
    st.image(Image.open("./images/icon.png"))

# Add sidebar success message.
st.sidebar.success("Select a topic above.")

