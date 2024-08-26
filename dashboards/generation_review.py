import os
import re
import json
import tempfile
from contextlib import nullcontext

import torch
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from omegaconf import OmegaConf, DictConfig

from model.gpt2 import GPT, GPTConfig
from model.dummy import RepeatingModel
import generation.generators as generators
from generation.tasks import Task, task_map
from dashboards.components import download_button
import database.database_manager as database_manager
from dashboards.utils import dataset_configuration, select_model_and_device
from model.tokenizers import AwesomeTokenizer, ExponentialTokenizer, special_tokens


def load_cfg(checkpoint: dict) -> DictConfig:
    train_config = checkpoint["config"]
    return OmegaConf.create(train_config)


def load_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_tokenizer(cfg: DictConfig):
    if "tokenizer" in cfg:
        tokenizer_parameters = OmegaConf.to_container(cfg.tokenizer.tokenizer_parameters)
        tokenizer_parameters |= {"special_tokens": special_tokens}

        if cfg.tokenizer.tokenizer == "AwesomeMidiTokenizer":
            min_time_unit = tokenizer_parameters["min_time_unit"]
            n_velocity_bins = tokenizer_parameters["min_velocity_bins"]
            tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
            return AwesomeTokenizer.from_file(tokenizer_path)
        else:
            return ExponentialTokenizer(**tokenizer_parameters)
    else:
        tokenizer_parameters = OmegaConf.to_container(cfg.data.tokenizer_parameters)
        tokenizer_parameters |= {"special_tokens": special_tokens}

        if cfg.data.tokenizer == "AwesomeMidiTokenizer":
            min_time_unit = tokenizer_parameters["min_time_unit"]
            n_velocity_bins = tokenizer_parameters["min_velocity_bins"]
            tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
            return AwesomeTokenizer.from_file(tokenizer_path)
        else:
            return ExponentialTokenizer(**tokenizer_parameters)


def initialize_gpt_model(
    cfg: DictConfig,
    checkpoint: dict,
    device: torch.device,
    pad_token_id: int = 0,
) -> GPT:
    """
    Initializes the GPT model using the given configurations and checkpoint.

    Parameters:
        cfg (DictConfig): The configuration object.
        dataset_config (dict): The dataset configuration.
        checkpoint (dict): The model checkpoint.
        device (torch.device): The device to load the model on.

    Returns:
        GPT: The initialized GPT model.
    """
    model_args = {
        "n_layer": cfg.model.n_layer,
        "n_head": cfg.model.n_head,
        "n_embd": cfg.model.n_embd,
        "block_size": cfg.data.sequence_length,
        "bias": cfg.model.bias,
        "vocab_size": None,
        "dropout": cfg.model.dropout,
    }

    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, pad_token_id=pad_token_id)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


def select_generator():
    st.header("Generation Configuration")
    generator_name = st.selectbox(label="Generator", options=generators.generator_types.keys())

    if "NextToken" in generator_name:
        tasks = ["next_token_prediction"]
    else:
        tasks = list(task_map.keys())
    task = st.sidebar.selectbox("Task", tasks)
    parameters = {"task": task}

    for parameter, value in generators.generator_types[generator_name].default_parameters().items():
        parameters |= {parameter: st.sidebar.number_input(label=parameter, value=value)}

    generator = generators.MidiGenerator.get_generator(generator_name=generator_name, parameters=parameters)

    return generator


def prepare_prompt(task: str, notes: pd.DataFrame, start_note_id: int = None, end_note_id: int = None):
    if start_note_id is not None and end_note_id is not None:
        notes = notes.iloc[start_note_id : end_note_id + 1].reset_index(drop=True)
    offset = notes.start.min()
    notes.start -= offset
    notes.end -= offset
    if task == "next_token_prediction":
        return notes, pd.DataFrame(columns=notes.columns)
    task_generator = Task.get_task(task_name=task)
    prompt_notes, target_notes = task_generator.generate(notes=notes)
    return prompt_notes, target_notes


def upload_midi_file(task: str):
    st.header("Upload Custom MIDI")
    uploaded_file = st.file_uploader("Choose a MIDI file", type="mid")
    use_whole_file = st.checkbox("Use whole file as prompt (skip preprocessing)", value=False)

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_midi_file:
            temp_midi_file.write(uploaded_file.getvalue())
            temp_midi_path = temp_midi_file.name

        try:
            midi_file = ff.MidiFile(temp_midi_path)
            notes = midi_file.piece.df

            st.success("MIDI file uploaded successfully!")
            streamlit_pianoroll.from_fortepyan(piece=ff.MidiPiece(notes))

            # Add note range selection
            note_id_cols = st.columns(2)
            start_note_id = note_id_cols[0].number_input(
                "Start Note ID",
                min_value=0,
                max_value=len(notes) - 1,
                value=0,
            )
            end_note_id = note_id_cols[1].number_input(
                "End Note ID",
                min_value=start_note_id,
                max_value=len(notes) - 1,
                value=len(notes) - 1,
            )

            if not use_whole_file:
                notes, _ = prepare_prompt(
                    task=task,
                    notes=notes,
                    start_note_id=start_note_id,
                    end_note_id=end_note_id,
                )
            else:
                notes = notes.iloc[start_note_id : end_note_id + 1].reset_index(drop=True)
            source = {
                "midi_name": temp_midi_path,
                "start": start_note_id,
                "end": end_note_id,
            }
            st.subheader("Selected Note Range")
            streamlit_pianoroll.from_fortepyan(piece=ff.MidiPiece(notes))

            return notes, source
        finally:
            os.unlink(temp_midi_path)

    return None, None


def main():
    st.title("MIDI Generation Dashboard")

    device, checkpoint_path = select_model_and_device()
    generator = select_generator()
    ctx = nullcontext()

    use_custom_midi = st.checkbox("Use custom MIDI file", value=False)

    if use_custom_midi:
        source_notes, source = upload_midi_file(task=generator.task)
        if source_notes is None:
            st.warning("Please upload a MIDI file to continue.")
            return
        # Source notes are the same as prompt notes because no extraction needed
        prompt_notes = source_notes.copy()

    else:
        dataset = dataset_configuration()
        prompt_index = st.number_input("Select prompt index", min_value=0, max_value=len(dataset) - 1, value=0)
        record = dataset[prompt_index]
        source_notes = pd.DataFrame(record["notes"])
        source = json.loads(record["source"])
        st.json(source)

        composer = source["composer"]
        title = source["title"]

        note_id_columns = st.columns(2)
        start_note_id = note_id_columns[0].number_input(
            "Start Note ID",
            min_value=0,
            max_value=len(source_notes) - 1,
            value=0,
        )
        end_note_id = note_id_columns[1].number_input(
            "End Note ID",
            min_value=start_note_id,
            max_value=len(source_notes) - 1,
            value=len(source_notes) - 1,
        )
        # Extract prompt by Task.generate
        prompt_notes, _ = prepare_prompt(
            task=generator.task,
            notes=source_notes,
            start_note_id=start_note_id,
            end_note_id=end_note_id,
        )
        source |= {
            "start": start_note_id,
            "end": end_note_id,
        }
        prompt_piece = ff.MidiPiece(prompt_notes)
        streamlit_pianoroll.from_fortepyan(piece=prompt_piece)

    if st.button("Generate"):
        if checkpoint_path == "DummyModel":
            model = RepeatingModel()
            tokenizer = ExponentialTokenizer(
                min_time_unit=0.01,
                n_velocity_bins=32,
                special_tokens=special_tokens,
            )
            model_name = "dummy"
        else:
            with st.spinner("Loading checkpoint..."):
                checkpoint = load_checkpoint(checkpoint_path, device="cpu")

            st.success(f"Model loaded! Best validation loss: {checkpoint['best_val_loss']:.4f}")
            if "wandb" in checkpoint:
                st.link_button(
                    label="View Training Run",
                    url=checkpoint["wandb"],
                )

            cfg = load_cfg(checkpoint=checkpoint)
            tokenizer = load_tokenizer(cfg=cfg)

            st.write("Training config")
            st.json(OmegaConf.to_container(cfg), expanded=False)

            model_name: str = os.path.basename(checkpoint_path)
            model_name = model_name.removesuffix(".pt")

            # Initialize model
            ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
            device_type = "cuda" if "cuda" in device else "cpu"
            ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
            model = initialize_gpt_model(cfg, checkpoint, device)

        with st.spinner("Generating MIDI..."):
            with ctx:
                prompt_notes, generated_notes = generator.generate(
                    prompt_notes=source_notes,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )
        st.dataframe(generated_notes)
        prompt_piece = ff.MidiPiece(df=prompt_notes.copy())
        generated_piece = ff.MidiPiece(df=generated_notes.copy())
        if generated_notes is not None and not generated_notes.empty:
            st.success("Generation complete!")
            streamlit_pianoroll.from_fortepyan(piece=generated_piece)

            if use_custom_midi:
                midi_name = f"{model_name}_custom_midi_variation"
            else:
                midi_name = f"{model_name}_variations_on_{title}_{composer}".lower()
            # Remove punctuation other than whitespace
            midi_name = re.sub(r"[^\w\s]", "", midi_name)
            generated_midi_path = f"tmp/generation_{midi_name}.mid"
            generated_piece.to_midi().write(generated_midi_path)
            with open(generated_midi_path, "rb") as file:
                st.markdown(
                    download_button(
                        file.read(),
                        generated_midi_path.split("/")[-1],
                        "Download generated midi",
                    ),
                    unsafe_allow_html=True,
                )
            os.unlink(generated_midi_path)

            streamlit_pianoroll.from_fortepyan(piece=prompt_piece, secondary_piece=generated_piece)

            def add_to_database():
                database_manager.insert_generation(
                    model_checkpoint=checkpoint,
                    model_name=model_name,
                    generator=generator,
                    generated_notes=generated_notes,
                    prompt_notes=prompt_notes,
                    source_notes=source_notes,
                    source=source,
                )

            st.button(
                "Add to database",
                key="add",
                on_click=add_to_database,
            )

            out_piece = ff.MidiPiece(pd.concat([prompt_notes, generated_notes]))
            # Allow download of the full MIDI with context
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
        else:
            st.error("Generation failed or not implemented.")


if __name__ == "__main__":
    main()
