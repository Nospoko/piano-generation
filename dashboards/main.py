import streamlit as st

from dashboards.browse_database import main as browse_dataset
from dashboards.generator_review import main as generator_review
from dashboards.generation_review import main as generation_review


def main():
    dashboards = ["browse_dataset", "generation_review", "generator_review"]
    dashboard = st.selectbox(label="Select dashboard", options=dashboards)

    match dashboard:
        case "browse_dataset":
            browse_dataset()
        case "generator_review":
            generator_review()
        case "generation_review":
            generation_review()


if __name__ == "__main__":
    main()
