import streamlit as st

from dashboards.task_review import main as task_review
from dashboards.browse_database import main as browse_dataset
from dashboards.transcription_human_comparison import main as comparison

st.set_page_config(
    page_title="PyData Presentation",
    page_icon=":musical_keyboard:",
    layout="wide",
)


def main():
    dashboards = ["browse_dataset", "generation_review", "task_review", "transcription human comparison"]
    dashboard = st.selectbox(label="Select dashboard", options=dashboards)

    match dashboard:
        case "browse_dataset":
            browse_dataset()
        case "task_review":
            task_review()
        case "transcription human comparison":
            comparison()


if __name__ == "__main__":
    main()
