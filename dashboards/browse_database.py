import os

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from streamlit.errors import DuplicateWidgetID

from dashboards.components import download_button
import database.database_manager as database_manager


def format_model_params(model_params):
    total_tokens, model_loss = model_params
    return f"{total_tokens:,}, best_val_loss: {model_loss}"


def main():
    st.title("MIDI Transformers Database Browser")

    tab1, tab2, tab3, tab4 = st.tabs(["Model Predictions", "Models", "Generators Parameters", "Prompt Notes"])

    with tab1:
        st.header("Model Predictions")

        models_df = database_manager.select_models_with_generations()
        model_names = models_df["name"].unique().tolist()

        selected_model_name = st.selectbox("Select Model", model_names, key="model")

        if selected_model_name:
            selected_models = models_df[models_df["name"] == selected_model_name]
            model_tokens = selected_models["total_tokens"].tolist()
            model_losses = selected_models["best_val_loss"].tolist()

            selected_model_params = st.selectbox(
                label="Select tokens",
                options=zip(model_tokens, model_losses),
                format_func=format_model_params,
            )
            selected_model_tokens, _ = selected_model_params
            selected_model = selected_models[selected_models["total_tokens"] == selected_model_tokens].iloc[0]
            st.json(selected_model.to_dict(), expanded=False)

            if pd.notna(selected_model["wandb_link"]):
                st.link_button("View Model on W&B", url=selected_model["wandb_link"])
            else:
                st.write("No W&B link available for this model")

            selected_model_id = selected_model["model_id"]

            # Get unique tasks for this model's predictions
            tasks = database_manager.get_model_tasks(model_id=selected_model_id)
            selected_task = st.selectbox("Select Task", tasks + ["All"], key="task_selector")
            # Filter predictions based on selected task
            if selected_task != "All":
                generator_filters = {"task": selected_task}
            else:
                generator_filters = None

            # Fetch predictions for the selected model and task
            predictions_df = database_manager.get_model_predictions(
                model_filters={"model_id": selected_model_id},
                generator_filters=generator_filters,
            )

            if not predictions_df.empty:
                for _, row in predictions_df.iterrows():
                    # generator = database_manager.get_generator(row["generator_id"]).iloc[0].to_dict()
                    prompt_notes = row["prompt_notes"]
                    prompt_notes_df = pd.DataFrame(prompt_notes)

                    generated_notes = row["generated_notes"]
                    generated_notes_df = pd.DataFrame(generated_notes)
                    generated_piece = ff.MidiPiece(df=generated_notes_df)
                    prompt_piece = ff.MidiPiece(df=prompt_notes_df)

                    st.write("#### Prompt")
                    try:
                        streamlit_pianoroll.from_fortepyan(piece=prompt_piece)
                    except DuplicateWidgetID:
                        st.write("Duplicate widget")
                    st.write("#### Generated")
                    try:
                        streamlit_pianoroll.from_fortepyan(piece=generated_piece)
                    except DuplicateWidgetID:
                        st.write("Duplicate widget")

                    # Allow download of the generated MIDI with
                    midi_name = f"{selected_model_name}_{selected_model_tokens:.2f}_generation"
                    midi_path = f"tmp/{midi_name}.mid"
                    generated_piece.to_midi().write(midi_path)
                    with open(midi_path, "rb") as file:
                        st.markdown(
                            download_button(
                                file.read(),
                                midi_path.split("/")[-1],
                                "Download midi with context",
                            ),
                            unsafe_allow_html=True,
                        )
                    os.unlink(midi_path)
                    st.write("#### Together")

                    try:
                        streamlit_pianoroll.from_fortepyan(piece=prompt_piece, secondary_piece=generated_piece)
                    except DuplicateWidgetID:
                        st.write("Duplicate widget")

                    out_piece = ff.MidiPiece(pd.concat([prompt_notes_df, generated_notes_df]))
                    # Allow download of the full MIDI with context\
                    midi_name = f"{selected_model_name}_{selected_model_tokens:.2f}_variations"
                    full_midi_path = f"tmp/{midi_name}.mid"
                    out_piece.to_midi().write(full_midi_path)
                    with open(full_midi_path, "rb") as file:
                        st.markdown(
                            download_button(
                                file.read(),
                                full_midi_path.split("/")[-1],
                                "Download midi with context",
                            ),
                            unsafe_allow_html=True,
                        )
                    os.unlink(full_midi_path)
                    st.divider()  # Add a divider between predictions

                else:
                    st.write("No predictions found for this prompt and model combination.")

    with tab2:
        st.header("Models")
        models_df = database_manager.select_models_with_generations()
        st.write(models_df)

        st.subheader("Purge Model")
        model_to_purge = st.selectbox("Select a model to purge", models_df["name"].tolist())
        if st.button("Purge Selected Model"):
            try:
                database_manager.purge_model(model_to_purge)
                st.success(f"Model '{model_to_purge}' has been purged successfully.")
            except Exception as e:
                st.error(f"An error occurred while purging the model: {str(e)}")

    with tab3:
        st.header("Generators")
        parameters_df = database_manager.get_all_generators()
        st.write(parameters_df)

    with tab4:
        st.header("Prompt Notes")
        prompts_df = database_manager.get_all_prompt_notes()
        st.write(prompts_df)


if __name__ == "__main__":
    main()
