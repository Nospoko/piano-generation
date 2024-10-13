import os
import json
import tempfile

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from streamlit.errors import DuplicateWidgetID

from dashboards.utils import dataset_configuration
import piano_generation.database.database_manager as database_manager


def upload_midi_file():
    st.header("Upload Custom MIDI")
    uploaded_file = st.file_uploader("Choose a MIDI file", type="mid")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_midi_file:
            temp_midi_file.write(uploaded_file.getvalue())
            temp_midi_path = temp_midi_file.name

        try:
            midi_file = ff.MidiFile(temp_midi_path)
            notes = midi_file.piece.df

            st.success("MIDI file uploaded successfully!")
            streamlit_pianoroll.from_fortepyan(piece=ff.MidiPiece(notes))

            source = {"midi_name": uploaded_file.name, "custom_upload": True}

            return notes, source
        finally:
            os.unlink(temp_midi_path)

    return None, None


def select_maestro_piece():
    st.header("Select from Maestro Database")
    dataset = dataset_configuration()
    prompt_index = st.number_input("Select prompt index", min_value=0, max_value=len(dataset) - 1, value=0)
    record = dataset[prompt_index]
    source_notes = pd.DataFrame(record["notes"])
    source = json.loads(record["source"])
    st.json(source)

    composer = source["composer"]
    title = source["title"]

    st.subheader(f"{composer} - {title}")
    return source_notes, source


def prepare_notes_and_source(notes: pd.DataFrame, source: dict):
    note_id_columns = st.columns(2)
    start_note_id = note_id_columns[0].number_input(
        "Start Note ID",
        min_value=0,
        max_value=len(notes) - 1,
        value=0,
    )
    end_note_id = note_id_columns[1].number_input(
        "End Note ID",
        min_value=start_note_id,
        max_value=len(notes) - 1,
        value=len(notes) - 1,
    )

    selected_notes = notes.iloc[start_note_id : end_note_id + 1].reset_index(drop=True)
    offset = selected_notes.start.min()
    selected_notes.start -= offset
    selected_notes.end -= offset
    streamlit_pianoroll.from_fortepyan(piece=ff.MidiPiece(selected_notes))

    source |= {
        "start": int(start_note_id),
        "end": int(end_note_id),
    }

    return selected_notes, source


def add_source_to_database(notes: pd.DataFrame, source: dict):
    if st.button("Add to database"):
        source_id = database_manager.insert_source(
            notes=notes,
            source=source,
        )
        st.success(f"Added to database with source_id: {source_id}")


def view_source_from_database():
    st.header("View Source from Database")

    sources = database_manager.get_all_sources()
    source_ids = sources["source_id"].tolist()

    selected_source_id = st.selectbox("Select Source ID", source_ids)

    if selected_source_id:
        source_data = database_manager.get_source(source_id=selected_source_id)

        if not source_data.empty:
            source = source_data.iloc[0]
            st.json(source["source"])

            notes_df = pd.DataFrame(source["notes"])

            # Display the full piece
            full_piece = ff.MidiPiece(df=notes_df)
            st.write("#### Full MIDI")
            try:
                streamlit_pianoroll.from_fortepyan(piece=full_piece)
            except DuplicateWidgetID:
                st.write("Duplicate widget ID")
        else:
            st.warning("No data found for the selected source ID.")


def main():
    st.title("Source Management Dashboard")

    tab1, tab2, tab3 = st.tabs(["Add Custom Source", "Add Maestro Source", "View Source"])

    with tab1:
        st.header("Add Custom Source")
        notes, source = upload_midi_file()
        if notes is not None and source is not None:
            notes, source = prepare_notes_and_source(notes, source)
            add_source_to_database(notes, source)

    with tab2:
        st.header("Add Maestro Source")
        notes, source = select_maestro_piece()
        if notes is not None and source is not None:
            notes, source = prepare_notes_and_source(notes, source)
            add_source_to_database(notes, source)

    with tab3:
        view_source_from_database()


if __name__ == "__main__":
    main()
