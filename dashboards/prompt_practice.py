import os
from glob import glob

import torch
import requests
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

from piano_generation import MidiGenerator
from dashboards.components import download_button
from piano_generation.utils import load_cfg, load_tokenizer, load_checkpoint, initialize_gpt_model


def main():
    # See the readme to figure out how you can get this checkpoint
    checkpoint_path = "checkpoints/midi-gpt2-302M-subsequence-4096-ctx-2024-09-08-19-42last.pt"
    st.write("Checkpoint:", checkpoint_path)
    checkpoint = load_cache_checkpoint(checkpoint_path)

    random_seed = st.number_input(
        label="random seed",
        value=137,
        max_value=100_000,
        min_value=0,
    )
    max_new_tokens = st.number_input(
        label="max new tokens",
        min_value=64,
        max_value=4096,
        value=2048,
    )
    temperature = st.number_input(
        label="temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.05,
    )

    speedup_factor = st.number_input(
        label="speedup factor",
        value=1.2,
        min_value=0.3,
        max_value=2.5,
    )

    # TODO: What would be a convenient way to manage prompts
    # for a user? Definitely needs an upload
    prompt_options = glob("tmp/*.mid") + [None]

    task_options = [
        "above_median_prediction",
        "above_high_quartile_prediction",
        "below_median_prediction",
        "middle_quartiles_prediction",
        "extreme_quartiles_prediction",
    ]
    task_name = st.selectbox(
        label="select task name",
        options=task_options,
        index=0,
    )
    prompt_path = st.selectbox(
        label="select prompt file",
        options=prompt_options,
        index=None,
    )
    pianoroll_apikey = st.text_input(
        label="pianoroll apikey",
        type="password",
    )
    st.write(pianoroll_apikey)

    if not prompt_path:
        return

    prompt_piece = ff.MidiPiece.from_file(prompt_path)
    prompt_notes: pd.DataFrame = prompt_piece.df
    prompt_notes.start *= speedup_factor
    prompt_notes.end *= speedup_factor

    streamlit_pianoroll.from_fortepyan(prompt_piece)
    #
    # Create a generator
    generator = MidiGenerator.get_generator(
        generator_name="StaticGenerator",
        parameters={
            "task": task_name,
            "notes_in_prompt": 128,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
    )
    model = checkpoint["model"]
    tokenizer = checkpoint["tokenizer"]

    possible_tokens = ["<BACH>", "<MOZART>", "<CHOPIN>", None]
    for additional_token in possible_tokens:
        st.write("Token:", additional_token, "task:", task_name)
        # Generator randomness comes from torch.multinomial, so we can make it
        # fully deterministic by setting global torch random seed
        for it in range(6):
            local_seed = random_seed + it * 1000
            st.write("Seed:", local_seed)

            # This acts as a caching key
            generation_properties = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "speedup_factor": speedup_factor,
                "prompt_path": prompt_path,
                "task_name": task_name,
                "local_seed": local_seed,
            }

            additional_tokens = [additional_token] if additional_token else []
            prompt_notes, generated_notes = cache_generation(
                prompt_notes=prompt_notes,
                seed=local_seed,
                _model=model,
                _generator=generator,
                _tokenizer=tokenizer,
                device="cuda",
                additional_tokens=additional_tokens,
                **generation_properties,
            )

            prompt_piece = ff.MidiPiece(prompt_notes)
            generated_piece = ff.MidiPiece(generated_notes)

            streamlit_pianoroll.from_fortepyan(prompt_piece, generated_piece)

            if pianoroll_apikey:
                # TODO: Add title and description control
                make_proll_post = st.button(
                    label="post to pianoroll.io",
                    key=f"{it}-{additional_token}",
                )
                if make_proll_post:
                    post_to_pianoroll(
                        model_piece=generated_piece,
                        prompt_piece=prompt_piece,
                        pianoroll_apikey=pianoroll_apikey,
                    )
                    st.write("POSTED!")

            out_piece = ff.MidiPiece(pd.concat([prompt_notes, generated_notes]))
            # Allow download of the full MIDI with context
            full_midi_path = f"tmp/tmp_{additional_token}_{local_seed}.mid"
            out_piece.to_midi().write(full_midi_path)
            with open(full_midi_path, "rb") as file:
                st.markdown(
                    download_button(
                        object_to_download=file.read(),
                        download_filename=full_midi_path.split("/")[-1],
                        button_text="Download midi with context",
                    ),
                    unsafe_allow_html=True,
                )
            st.write("---")
    os.unlink(full_midi_path)


def post_to_pianoroll(
    model_piece: ff.MidiPiece,
    prompt_piece: ff.MidiPiece,
    pianoroll_apikey: str,
):
    model_notes = model_piece.df.to_dict(orient="records")
    prompt_notes = prompt_piece.df.to_dict(orient="records")

    payload = {
        "model_notes": model_notes,
        "prompt_notes": prompt_notes,
        "post_title": "GENAI",
        "post_description": "My model did this!",
    }

    headers = {
        "UserApiToken": pianoroll_apikey,
    }
    api_endpoint = "https://pianoroll.io/api/v1/generation_pianorolls"
    r = requests.post(api_endpoint, headers=headers, json=payload)

    st.write(r)


@st.cache_data
def cache_generation(
    prompt_notes: pd.DataFrame,
    seed: int,
    _generator,
    _model,
    _tokenizer,
    additional_tokens: list[str],
    device: str = "cuda",
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    torch.random.manual_seed(seed)
    with st.spinner("gpu goes brrrrrrrrrr"):
        prompt_notes, generated_notes = _generator.generate(
            prompt_notes=prompt_notes,
            model=_model,
            tokenizer=_tokenizer,
            device="cuda",
            additional_tokens=additional_tokens,
        )

    return prompt_notes, generated_notes


@st.cache_data
def load_cache_checkpoint(checkpoint_path: str):
    # Load a pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    cfg = load_cfg(checkpoint)
    tokenizer = load_tokenizer(cfg)
    model = initialize_gpt_model(cfg, checkpoint, device=device)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "cfg": cfg,
    }


if __name__ == "__main__":
    main()
