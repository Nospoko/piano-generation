import streamlit as st

from dashboards.generator_review import main as generator_review


def main():
    dashboards = ["generator_review"]
    dashboard = st.selectbox(label="Select dashboard", options=dashboards)

    match dashboard:
        case "generator_review":
            generator_review()


if __name__ == "__main__":
    main()
