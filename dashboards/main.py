import streamlit as st

from dashboards.task_review import main as task_review
from dashboards.browse_database import main as browse_dataset
from dashboards.generation_review import main as generation_review

st.set_page_config(
    page_title="Piano Generation",
    page_icon=":musical_keyboard:",
    layout="wide",
)


def main():
    dashboards = ["browse_dataset", "generation_review", "task_review"]
    dashboard = st.selectbox(label="Select dashboard", options=dashboards)

    match dashboard:
        case "browse_dataset":
            browse_dataset()
        case "task_review":
            task_review()
        case "generation_review":
            generation_review()


if __name__ == "__main__":
    main()
