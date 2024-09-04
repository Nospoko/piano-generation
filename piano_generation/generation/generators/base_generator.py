from abc import ABC, abstractmethod

import torch
import pandas as pd
from torch import nn

from piano_generation import AwesomeTokenizer, ExponentialTokenizer
from piano_generation.generation.generators.generator_types import generator_types


class MidiGenerator(ABC):
    def __init__(self, task: str):
        self.task = task

    @abstractmethod
    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
        additional_tokens: list[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @staticmethod
    def get_generator(generator_name: str, parameters) -> "MidiGenerator":
        return generator_types.get(generator_name)(**parameters)

    @staticmethod
    def default_parameters() -> dict:
        return {
            "task": "next_token_prediction",
        }

    @property
    def parameters(self) -> dict:
        return {}
