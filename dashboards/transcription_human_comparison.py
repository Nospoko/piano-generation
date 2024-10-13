import fortepyan as ff
import streamlit as st
import streamlit_pianoroll


def main():
    transcription_path = "tmp/scored_piece.mid"
    human_path = "tmp/scored_piece_human.mid"
    # side by side comparison
    display_columns = st.columns([1, 1])
    with display_columns[0]:
        prepared_piece = ff.MidiPiece.from_file(transcription_path)
        streamlit_pianoroll.from_fortepyan(piece=prepared_piece)
        st.dataframe(prepared_piece.df)
    with display_columns[1]:
        prepared_piece = ff.MidiPiece.from_file(human_path)
        streamlit_pianoroll.from_fortepyan(piece=prepared_piece)
        st.dataframe(prepared_piece.df)
    return
