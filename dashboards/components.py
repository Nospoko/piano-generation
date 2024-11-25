import re
import json
import uuid
import base64

import numpy as np
import pandas as pd
import streamlit as st
from piano_metrics import f1_piano, key_distribution, pitch_distribution


def initialize_metric_components() -> tuple[float, float, float, dict, dict, dict]:
    """Initialize metric calculation components for the dashboard"""
    st.header("Metrics Analysis")

    with st.expander("Metric Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Key Distribution")
            use_weighted_key = st.checkbox("Use weighted key", value=True)
            segment_duration = st.slider("Segment Duration (s)", 0.05, 0.5, 0.125, 0.025)

        with col2:
            st.subheader("Pitch Distribution")
            use_weighted_pitch = st.checkbox("Use weighted pitch", value=True)

        with col3:
            st.subheader("F1 Score")
            min_time_unit = st.number_input("Min Time Unit (s)", value=0.01, step=0.001, format="%.3f")
            velocity_threshold = st.number_input("Velocity Threshold", value=30, step=1)
            use_pitch_class = st.checkbox("Use pitch class", value=True)

    def calculate_metrics(prompt_df: pd.DataFrame, generated_df: pd.DataFrame) -> tuple[float, float, float, dict, dict, dict]:
        key_corr, key_metrics = key_distribution.calculate_key_correlation(
            target_df=prompt_df, generated_df=generated_df, segment_duration=segment_duration, use_weighted=use_weighted_key
        )

        pitch_corr, pitch_metrics = pitch_distribution.calculate_pitch_correlation(
            target_df=prompt_df, generated_df=generated_df, use_weighted=use_weighted_pitch
        )

        f1_score, f1_metrics = f1_piano.calculate_f1(
            target_df=prompt_df,
            generated_df=generated_df,
            min_time_unit=min_time_unit,
            velocity_threshold=velocity_threshold,
            use_pitch_class=use_pitch_class,
        )

        return key_corr, pitch_corr, f1_score, key_metrics, pitch_metrics, f1_metrics

    return calculate_metrics


def display_metrics(
    key_corr: float, pitch_corr: float, f1_score: float, key_metrics: dict, pitch_metrics: dict, f1_metrics: dict
):
    """Display metric results in the dashboard"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Key Distribution Correlation", f"{key_corr:.3f}")
        key_df = pd.DataFrame(
            {
                "Target": key_metrics["target_top_keys"][:3],
                "Generated": key_metrics["generated_top_keys"][:3],
            }
        )
        st.dataframe(key_df.style.set_caption("Top 3 Keys"))

    with col2:
        st.metric("Pitch Distribution Correlation", f"{pitch_corr:.3f}")
        pitch_df = pd.DataFrame(
            {
                "Metric": ["Active Pitches"],
                "Target": [pitch_metrics["target_active_pitches"]],
                "Generated": [pitch_metrics["generated_active_pitches"]],
            }
        )
        st.dataframe(pitch_df)

    with col3:
        st.metric("F1 Score", f"{f1_score:.3f}")
        f1_df = pd.DataFrame(
            {
                "Metric": ["Precision", "Recall"],
                "Value": [f"{np.mean(f1_metrics['precision']):.3f}", f"{np.mean(f1_metrics['recall']):.3f}"],
            }
        )
        st.dataframe(f1_df)


def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if isinstance(object_to_download, bytes):
        pass

    elif isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub(r"\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    a_html = f"""
    <a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">
        {button_text}
    </a>
    <br></br>
    """
    button_html = custom_css + a_html

    return button_html
