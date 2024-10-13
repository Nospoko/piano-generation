import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset

from piano_generation.generation.tasks import Task, task_map


@st.cache_data()
def load_hf_dataset():
    return load_dataset("roszcz/maestro-sustain-v2", split="test")


def main():
    dataset = load_hf_dataset()
    record__id = st.number_input(label="record_id", value=100)
    record = dataset[record__id]

    piece = ff.MidiPiece.from_huggingface(record=record)
    piece = piece[:256]

    task_names = task_map.keys()
    task_name = st.selectbox(label="task name", options=task_names)
    task_generator = Task.get_task(task_name=task_name)

    prompt_notes, target_notes = task_generator.generate(notes=piece.df.copy())
    prompt_piece = ff.MidiPiece(prompt_notes)
    target_piece = ff.MidiPiece(target_notes)

    streamlit_pianoroll.from_fortepyan(piece=prompt_piece, secondary_piece=target_piece)
    pianoroll_columns = st.columns(2)

    with pianoroll_columns[0]:
        streamlit_pianoroll.from_fortepyan(piece=prompt_piece)
    with pianoroll_columns[1]:
        streamlit_pianoroll.from_fortepyan(piece=target_piece)


if __name__ == "__main__":
    main()
