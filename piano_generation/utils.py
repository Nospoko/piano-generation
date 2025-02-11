import torch
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import ExponentialTimeTokenizer
from midi_trainable_tokenizers import AwesomeMidiTokenizer

from piano_generation import GPT
from piano_generation.artifacts import special_tokens


def load_cfg(checkpoint: dict) -> DictConfig:
    train_config = checkpoint["config"]
    return OmegaConf.create(train_config)


def load_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_tokenizer(cfg: DictConfig):
    if "tokenizer" in cfg:
        if "tokenizer_parameters" in cfg.tokenizer:
            tokenizer_parameters = OmegaConf.to_container(cfg.tokenizer.tokenizer_parameters)
        else:
            tokenizer_parameters = OmegaConf.to_container(cfg.tokenizer.parameters)
        tokenizer_parameters |= {"special_tokens": special_tokens}
        if "name" in cfg.tokenizer:
            name = cfg.tokenizer.name
        elif "tokenizer" in cfg.tokenizer:
            name = cfg.tokenizer.tokenizer
        if name == "AwesomeMidiTokenizer":
            min_time_unit = tokenizer_parameters["min_time_unit"]
            n_velocity_bins = tokenizer_parameters["min_velocity_bins"]
            tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
            return AwesomeMidiTokenizer.from_file(tokenizer_path)
        else:
            return ExponentialTimeTokenizer(**tokenizer_parameters)
    else:
        tokenizer_parameters = OmegaConf.to_container(cfg.data.tokenizer_parameters)
        tokenizer_parameters |= {"special_tokens": special_tokens}

        if cfg.data.tokenizer == "AwesomeMidiTokenizer":
            min_time_unit = tokenizer_parameters["min_time_unit"]
            n_velocity_bins = tokenizer_parameters["min_velocity_bins"]
            tokenizer_path = f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
            return AwesomeMidiTokenizer.from_file(tokenizer_path)
        else:
            return ExponentialTimeTokenizer(**tokenizer_parameters)


def initialize_gpt_model(
    cfg: DictConfig,
    checkpoint: dict,
    device: torch.device,
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
    tokenizer = ExponentialTimeTokenizer.from_dict(checkpoint["tokenizer_desc"])

    model = GPT(
        config=cfg.model,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=tokenizer.vocab_size,
    )
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model
