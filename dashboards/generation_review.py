import os
import re
import json
from glob import glob
from contextlib import nullcontext

import yaml
import torch
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig

from model.gpt2 import GPT, GPTConfig
import generation.generators as generators
from generation.tasks import Task, task_map
from dashboards.components import download_button
from model.tokenizers import AwesomeTokenizer, ExponentialTokenizer, special_tokens


def load_cfg(checkpoint: dict) -> DictConfig:
    train_config = checkpoint["config"]
    return OmegaConf.create(train_config)


def select_model_and_device():
    with st.sidebar:
        st.header("Model Configuration")
        devices = [f"cuda:{it}" for it in range(torch.cuda.device_count())] + ["cpu"]
        device = st.selectbox("Select Device", options=devices, help="Choose the device to run the model on")
        checkpoint_path = st.selectbox(
            "Select Checkpoint",
            options=glob("checkpoints/*.pt"),
            help="Choose the model checkpoint to use",
        )

    return device, checkpoint_path


def select_part_dataset(midi_dataset: Dataset) -> Dataset:
    """
    Allows the user to select a part of the dataset based on composer and title.

    Parameters:
        midi_dataset (Dataset): The MIDI dataset to select from.

    Returns:
        Dataset: The selected part of the dataset.
    """
    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]
    source_df["title"] = [source["title"] for source in source_df.source]

    composers = source_df.composer.unique()
    selected_composer = st.selectbox(
        "Select composer",
        options=composers,
        index=3,
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox("Select title", options=piece_titles)

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    part_df = source_df[ids]
    part_dataset = midi_dataset.select(part_df.index.values)

    return part_dataset


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


def dataset_configuration():
    st.header("Dataset Configuration")
    col1, col2 = st.columns(2)
    with col1:
        dataset_path = st.text_input(
            "Dataset Path",
            value="roszcz/maestro-sustain-v2",
            help="Enter the path to the dataset",
        )
    with col2:
        dataset_split = st.selectbox(
            "Dataset Split",
            options=["validation", "train", "test"],
            help="Choose the dataset split to use",
        )

    with st.spinner("Loading dataset..."):
        dataset = load_hf_dataset(dataset_path=dataset_path, dataset_split=dataset_split)
        dataset = select_part_dataset(midi_dataset=dataset)

    st.success(f"Dataset loaded! Total records: {len(dataset)}")
    return dataset


@st.cache_data
def load_hf_dataset(dataset_path: str, dataset_split: str):
    dataset = load_dataset(
        dataset_path,
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
    )
    return dataset


def select_generator():
    st.header("Generation Configuration")
    generator_type = st.selectbox(label="Generator", options=generators.generator_types.keys())

    if generator_type == "NextTokenGenerator":
        task = st.sidebar.selectbox("Task", ["next_token_prediction"] + list(task_map.keys()))
        prompt_context_duration = st.sidebar.number_input("Prompt Context Duration", value=20)
        max_new_tokens = st.sidebar.number_input("Max New Tokens", value=1024)
        temperature = st.sidebar.number_input("Temperature", value=1.0)
        generator = generators.NextTokenGenerator(
            prompt_context_duration=prompt_context_duration,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    else:
        task = st.sidebar.selectbox("Task", list(task_map.keys()))
        prompt_context_duration = st.sidebar.number_input("Prompt Context Duration", value=15.0)
        target_context_duration = st.sidebar.number_input("Target Context Duration", value=10.0)
        time_step = st.sidebar.number_input("Time Step", value=5.0)
        temperature = st.sidebar.number_input("Temperature", value=1.0)
        max_new_tokens = st.sidebar.number_input("Max New Tokens", value=1024)

        generator = generators.generator_types[generator_type](
            task=task,
            prompt_context_duration=prompt_context_duration,
            target_context_duration=target_context_duration,
            time_step=time_step,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    return generator


def prepare_prompt(
    task: str,
    notes: pd.DataFrame,
):
    task_generator = Task.get_task(task_name=task)
    prompt_notes, target_notes = task_generator.generate(notes=notes)
    return prompt_notes, target_notes


def main():
    st.title("MIDI Generation Dashboard")

    # Load model and dataset
    device, checkpoint_path = select_model_and_device()
    dataset = dataset_configuration()

    # Generation configuration
    generator = select_generator()

    # Select a prompt from the dataset
    prompt_index = st.number_input("Select prompt index", min_value=0, max_value=len(dataset) - 1, value=0)
    record = dataset[prompt_index]
    prompt_notes = record["notes"]
    source = json.loads(record["source"])
    st.json(source)

    composer = source["composer"]
    title = source["title"]

    prompt_notes = pd.DataFrame(prompt_notes)
    prompt_notes, _ = prepare_prompt(
        task=generator.task,
        notes=prompt_notes,
    )
    prompt_piece = ff.MidiPiece(prompt_notes)
    streamlit_pianoroll.from_fortepyan(piece=prompt_piece)

    if st.button("Generate"):
        with st.spinner("Loading checkpoint..."):
            checkpoint = load_checkpoint(checkpoint_path, device="cpu")

        st.success(f"Model loaded! Best validation loss: {checkpoint['best_val_loss']:.4f}")
        if "wandb" in checkpoint:
            st.link_button(label="View Training Run", url=checkpoint["wandb"])

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
                    prompt_notes=prompt_notes,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )

        prompt_piece = ff.MidiPiece(df=prompt_notes)
        generated_piece = ff.MidiPiece(df=generated_notes)
        if generated_notes:
            st.success("Generation complete!")
            streamlit_pianoroll.from_fortepyan(piece=generated_piece)
            streamlit_pianoroll.from_fortepyan(piece=prompt_piece, secondary_piece=generated_piece)

            out_piece = ff.MidiPiece(pd.concat([prompt_notes, generated_notes]))
            # Allow download of the full MIDI with context\
            midi_name = f"{model_name}_variations_on_{title}_{composer}".lower()
            midi_name = re.sub(r"[^\w\s]", "", midi_name)
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
