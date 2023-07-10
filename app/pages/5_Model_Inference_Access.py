"""
Web application real-time inference page via streamlit.
"""

import pandas as pd
from PIL import Image
import streamlit as st


from birdwatcher.ml.inference import get_trained_inference_pipeline


INTRO_MARKDOWN = """
You can access the underlying model we used to generate sentiment
classification scores either at the level of a single value or in batch.
If scoring in batch, make sure to specify the name of the
text-containing column as 'text'!
"""

ONLINE_INPUT_LABEL = """
Enter text for sentiment classification scoring.
"""

ONLINE_RESULT_MARKDOWN = """
The text entered above was scored as having a {sentiment} sentiment.
"""

UPLOAD_LABEL = """
Provide a csv file containing a column, 'text', of strings to score.
"""

DOWNLOAD_LABEL = """
Download dataset scores.
"""


@st.cache_data
def perform_batch_inference(uploaded_file):
    df_upload = pd.read_csv(uploaded_file)
    inference_pipeline = get_trained_inference_pipeline(
        data_key="covid",
        save=False,
        verbose=False
    )
    df_preds = pd.DataFrame({
        "preds": inference_pipeline.predict(df_upload)
    })
    return pd.concat(
        (df_upload, df_preds),
        axis="columns"
    )


@st.cache_data
def perform_online_inference(text: str):
    df_text = pd.DataFrame({
        "text": [text]
    })
    inference_pipeline = get_trained_inference_pipeline(
        data_key="covid",
        save=False,
        verbose=False
    )
    df_preds = pd.DataFrame({
        "preds": inference_pipeline.predict(df_text)
    })
    sentiment = "positive" if df_preds.preds.iloc[0] == 1 else "negative"
    return ONLINE_RESULT_MARKDOWN.format(sentiment=sentiment)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Let's go Birdwatching",
        page_icon="./pages/images/icon.png"
    )
    
    # Create a title.
    st.title("Model Inference Access")
    st.sidebar.success("Select a topic above.")
    
    col1, col2 = st.columns([7, 3])
    with col1:
        # Display intro markdown.
        st.write(INTRO_MARKDOWN)

        # Display online inference widgets.
        with st.form("online_inference_form"):
            text = st.text_input(label=ONLINE_INPUT_LABEL)
            submit_button = st.form_submit_button(label="Submit")

        if ((text is not None) and submit_button):
            st.write(perform_online_inference(text))
        
        # Display batch inference forms/widgets.
        with st.form(key="batch_inference_form"):
            uploaded_file = st.file_uploader(
                label=UPLOAD_LABEL,
                type="csv"
            )
            submit_button = st.form_submit_button(label="Submit")

        if ((uploaded_file is not None) and submit_button):
            df_return = perform_batch_inference(uploaded_file)
            st.download_button(
                label=DOWNLOAD_LABEL,
                data=df_return.to_csv(index=False).encode('utf-8'),
                file_name="sentiment_scores.csv",
                mime="text/csv"
            )

    with col2:
        st.image(
            Image.open("./pages/images/robot_reading_ai_generated.png"),
            caption="Image created using hotpot ai art generator."
        )