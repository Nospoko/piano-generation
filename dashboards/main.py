import streamlit as st

from dashboards.generator_review import main as generator_review
from dashboards.generation_review import main as generation_review


def main():
    dashboards = ["generation_review", "generator_review"]
    dashboard = st.selectbox(label="Select dashboard", options=dashboards)

    match dashboard:
        case "generator_review":
            generator_review()
        case "generation_review":
            generation_review()


if __name__ == "__main__":
    main()
