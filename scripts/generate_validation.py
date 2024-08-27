import os
import argparse
from contextlib import nullcontext

import torch
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from generation.tasks import Task
from model.gpt2 import GPT, GPTConfig
import generation.generators as generators
import database.database_manager as database_manager
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


def run_generation_step(
    model: GPT,
    checkpoint: dict,
    run_name: str,
    validation_examples: pd.DataFrame,
    task: Task,
    generator: generators.MidiGenerator,
    tokenizer: AwesomeTokenizer | ExponentialTokenizer,
    device: torch.device,
):
    for idx, example in validation_examples.iterrows():
        source_notes = pd.DataFrame(example["source_notes"])
        source = example["source"]

        prompt_notes = task.generate(notes=source_notes)

        prompt_notes, generation = generator.generate(
            prompt_notes=prompt_notes,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        database_manager.insert_generation(
            model_checkpoint=checkpoint,
            model_name=run_name,
            generated_notes=generation,
            prompt_notes=prompt_notes,
            source_notes=source_notes,
            source=source,
        )


def main(model_path: str, device: str, task: str):
    checkpoint = torch.load(f=model_path, map_location=device)

    cfg = load_cfg(checkpoint=checkpoint)
    tokenizer = load_tokenizer(cfg)

    model = initialize_gpt_model(
        cfg,
        checkpoint=checkpoint,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )

    generator = generators.SeqToSeqTokenwiseGenerator(
        task=task,
        prompt_context_length=1024,
        target_context_length=512,
        time_step=2,
        temperature=1.0,
        max_new_tokens=4096,
    )

    task = generator.task

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    validation_examples = database_manager.get_validation_sources()

    run_name = os.path.splitext(os.path.basename(model_path))[0]
    with ctx:
        run_generation_step(
            model=model,
            checkpoint=checkpoint,
            run_name=run_name,
            validation_examples=validation_examples,
            tokenizer=tokenizer,
            generator=generator,
            task=task,
            device=device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model generation on validation examples.")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("device", type=str, help="Device to perform calculations on.")
    parser.add_argument("task", type=str, help="Task for the model to perform generation.")
    args = parser.parse_args()

    main(args.model_path, args.device, args.task)
