import torch
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from midi_tokenizers import ExponentialTimeTokenizer

from piano_generation.artifacts import special_tokens
from piano_generation.generation.tasks import task_map
from piano_generation import DummyModel, NextTokenGenerator, SeqToSeqIterativeGenerator, SeqToSeqTokenwiseGenerator


def main():
    st.title("MIDI Generator Review Dashboard")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    generator_type = st.sidebar.selectbox(
        "Generator Type", ["NextTokenGenerator", "SeqToSeqTokenwiseGenerator", "SeqToSeqIterativeGenerator"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyModel().to(device)
    tokenizer = ExponentialTimeTokenizer(
        min_time_unit=0.01,
        n_velocity_bins=32,
        special_tokens=special_tokens,
    )
    # Generator-specific parameters
    if generator_type == "NextTokenGenerator":
        prompt_context_duration = st.sidebar.number_input("Prompt Context Duration", value=20)
        max_new_tokens = st.sidebar.number_input("Max New Tokens", value=1024)
        temperature = st.sidebar.number_input("Temperature", value=1.0)

        generator = NextTokenGenerator(
            prompt_context_duration=prompt_context_duration,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif generator_type in ["SeqToSeqTokenwiseGenerator", "SeqToSeqIterativeGenerator"]:
        task = st.sidebar.selectbox("Task", list(task_map.keys()))
        prompt_context_duration = st.sidebar.number_input("Prompt Context Duration", value=15.0)
        target_context_duration = st.sidebar.number_input("Target Context Duration", value=10.0)
        time_step = st.sidebar.number_input("Time Step", value=5.0)
        temperature = st.sidebar.number_input("Temperature", value=1.0)
        max_new_tokens = st.sidebar.number_input("Max New Tokens", value=1024)

        if generator_type == "SeqToSeqTokenwiseGenerator":
            generator = SeqToSeqTokenwiseGenerator(
                task=task,
                prompt_context_duration=prompt_context_duration,
                target_context_duration=target_context_duration,
                time_step=time_step,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        else:
            generator = SeqToSeqIterativeGenerator(
                task=task,
                prompt_context_duration=prompt_context_duration,
                target_context_duration=target_context_duration,
                time_step=time_step,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

    # Generate sample input
    st.header("Sample Input")
    num_notes = st.number_input("Number of input notes", value=30)
    input_piece = ff.MidiPiece.from_file("tmp/bach.mid")
    input_piece = input_piece[:num_notes]
    input_notes = input_piece.df

    st.write(input_notes)
    col1, col2 = st.columns(2)
    prompt_piece = ff.MidiPiece(input_notes)
    with col1:
        st.write("Prompt:")
        streamlit_pianoroll.from_fortepyan(piece=prompt_piece)

    # Generate output
    if st.button("Generate"):
        with st.spinner("Generating..."):
            prompt_notes, generated_notes = generator.generate(
                prompt_notes=input_notes,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )

        st.header("Generated Output")
        st.write(generated_notes)
        generated_piece = ff.MidiPiece(df=generated_notes)
        prompt_piece = ff.MidiPiece(df=prompt_notes)

        with col2:
            st.write("Generated:")
            streamlit_pianoroll.from_fortepyan(piece=generated_piece)

        # Visualize using streamlit_pianoroll
        st.header("Piano Roll Visualization")

        generated_piece = ff.MidiPiece(generated_notes)

        st.write("Combined View:")
        streamlit_pianoroll.from_fortepyan(piece=prompt_piece, secondary_piece=generated_piece)


if __name__ == "__main__":
    main()
