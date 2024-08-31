import re
from abc import ABC, abstractmethod

import torch
import pandas as pd
import streamlit as st
from torch import nn

from piano_generation import Task, AwesomeTokenizer, ExponentialTokenizer


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


class NextTokenGenerator(MidiGenerator):
    def __init__(
        self,
        task: str = "next_token_prediction",
        prompt_context_duration: float = 15.0,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
    ):
        super().__init__(task=task)
        self.prompt_context_duration = prompt_context_duration
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
    ):
        prompt_notes = prompt_notes[prompt_notes.end < self.prompt_context_duration]

        # Tokenize prompt notes
        input_sequence = tokenizer.tokenize(prompt_notes)

        # Convert tokens to ids and prepare input tensor
        input_token_ids = torch.tensor(
            [[tokenizer.token_to_id[token] for token in input_sequence]],
            device=device,
            dtype=torch.int64,
        )

        # Generate new tokens using the model
        with torch.no_grad():
            output = model.generate_new_tokens(
                input_token_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        # Convert output to numpy array and decode tokens
        output = output[0].cpu().numpy()
        out_tokens = [tokenizer.vocab[token_id] for token_id in output]

        # Convert tokens back to notes
        generated_notes = tokenizer.untokenize(out_tokens)
        generated_notes.start += prompt_notes.end.max()
        generated_notes.end += prompt_notes.end.max()
        return prompt_notes, generated_notes

    @staticmethod
    def default_parameters():
        return {
            "prompt_context_duration": 15.0,
            "max_new_tokens": 1024,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "prompt_context_duration": self.prompt_context_duration,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


class SeqToSeqIterativeGenerator(MidiGenerator):
    """
    This generation method performs calculations directly on tokens to keep the timing the same at all times.
    """

    def __init__(
        self,
        task: str,
        prompt_context_duration: float,
        target_context_duration: float,
        time_step: float,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ):
        super().__init__(task=task)
        self.prompt_context_duration = prompt_context_duration
        self.target_context_duration = target_context_duration
        self.time_step = time_step
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.task_generator = Task.get_task(task_name=task)

    @staticmethod
    def default_parameters() -> dict:
        return {
            "prompt_context_duration": 15.0,
            "target_context_duration": 10.0,
            "time_step": 5.0,
            "max_new_tokens": 1024,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "prompt_context_duration": self.prompt_context_duration,
            "target_context_duration": self.target_context_duration,
            "time_step": self.time_step,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    @staticmethod
    def calculate_token_duration(
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        tokens: list[str],
    ):
        t = 0
        for token in tokens:
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
        return t

    def trim_tokens_front(
        time_step: float,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        tokens: list[str],
    ):
        t = 0
        tokens = tokens.copy()
        for token in tokens:
            tokens.pop(0)
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
            if t >= time_step or len(tokens) == 0:
                break

        # in case too much was popped
        if t > time_step:
            dt = t - time_step
            time_tokens = tokenizer.tokenize_time_distance(dt=dt)
            tokens = time_tokens + tokens
        return tokens

    def trim_tokens_back(
        duration: float,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        tokens: list[str],
    ):
        t = 0
        result = []
        for token in tokens:
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
            result.append(token)
            if t >= duration:
                break
        return result

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        input_tokens = tokenizer.tokenize(prompt_notes)
        step_target_tokens = []
        output_tokens = []
        source_token = self.task_generator.source_token
        target_token = self.task_generator.target_token

        for _ in range(self.max_new_tokens):
            step_input_tokens = SeqToSeqTokenwiseGenerator.trim_tokens_back(
                duration=self.prompt_context_duration,
                tokenizer=tokenizer,
                tokens=input_tokens,
            )
            source_token = self.task_generator.source_token
            target_token = self.task_generator.target_token
            step_input_tokens = [source_token] + step_input_tokens + [target_token] + step_target_tokens

            step_token_ids = [tokenizer.token_to_id[token] for token in step_input_tokens]

            step_token_ids = torch.tensor(
                [step_token_ids],
                device=device,
                dtype=torch.int64,
            )
            next_token = model.generate_new_tokens(
                idx=step_token_ids,
                temperature=self.temperature,
                max_new_tokens=1,
            )

            next_token_id = next_token[0].cpu().numpy()[0]
            next_token = tokenizer.vocab[next_token_id]

            output_tokens.append(next_token)
            step_target_tokens.append(next_token)

            generated_notes_duration = self.calculate_token_duration(
                tokenizer=tokenizer,
                tokens=step_target_tokens,
            )
            # If the generated notes are longer than context_duration, move the generation window time_step to the right
            if generated_notes_duration > self.target_context_duration:
                input_tokens = SeqToSeqTokenwiseGenerator.trim_tokens_front(
                    time_step=self.time_step,
                    tokenizer=tokenizer,
                    tokens=input_tokens,
                )
                step_target_tokens = SeqToSeqTokenwiseGenerator.trim_tokens_front(
                    time_step=self.time_step,
                    tokenizer=tokenizer,
                    tokens=step_target_tokens,
                )

            if (
                SeqToSeqTokenwiseGenerator.calculate_token_duration(
                    tokenizer=tokenizer,
                    tokens=input_tokens,
                )
                == 0
            ):
                break
        target_notes = tokenizer.untokenize(tokens=output_tokens)
        return prompt_notes, target_notes


class NextTokenTokenwiseGenerator(MidiGenerator):
    def __init__(
        self,
        task: str = "next_token_prediction",
        prompt_context_length: float = 1024,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
    ):
        super().__init__(task=task)
        self.prompt_context_length = prompt_context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
    ):
        prompt_notes = prompt_notes[-self.prompt_context_length :]

        # Tokenize prompt notes
        input_sequence = tokenizer.tokenize(prompt_notes)

        # Convert tokens to ids and prepare input tensor
        input_token_ids = torch.tensor(
            [[tokenizer.token_to_id[token] for token in input_sequence]],
            device=device,
            dtype=torch.int64,
        )

        # Generate new tokens using the model
        with torch.no_grad():
            output = model.generate_new_tokens(
                input_token_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        # Convert output to numpy array and decode tokens
        output = output[0].cpu().numpy()
        out_tokens = [tokenizer.vocab[token_id] for token_id in output]

        # Convert tokens back to notes
        generated_notes = tokenizer.untokenize(out_tokens)
        generated_notes.start += prompt_notes.end.max()
        generated_notes.end += prompt_notes.end.max()
        return prompt_notes, generated_notes

    @staticmethod
    def default_parameters():
        return {
            "prompt_context_length": 1024,
            "max_new_tokens": 1024,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "prompt_context_length": self.prompt_context_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


class SeqToSeqTokenwiseGenerator(MidiGenerator):
    """
    This generation method performs calculations directly on tokens to keep the timing the same at all times.
    """

    def __init__(
        self,
        task: str,
        prompt_context_length: int,
        target_context_length: int,
        time_step: float,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ):
        super().__init__(task=task)
        self.prompt_context_length = prompt_context_length
        self.target_context_length = target_context_length
        self.time_step = time_step
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.task_generator = Task.get_task(task_name=task)

    @staticmethod
    def default_parameters() -> dict:
        return {
            "prompt_context_length": 1024,
            "target_context_length": 512,
            "time_step": 5.0,
            "max_new_tokens": 1024,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "prompt_context_length": self.prompt_context_length,
            "target_context_length": self.target_context_length,
            "time_step": self.time_step,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    @staticmethod
    def calculate_token_duration(
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        tokens: list[str],
    ):
        t = 0
        for token in tokens:
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
        return t

    @staticmethod
    def trim_tokens_front(
        time_step: float,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        tokens: list[str],
    ):
        t = 0
        tokens = tokens.copy()
        for token in tokens:
            tokens.pop(0)
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
            if t >= time_step or len(tokens) == 0:
                break

        # in case too much was popped
        if t > time_step:
            dt = t - time_step
            time_tokens = tokenizer.tokenize_time_distance(dt=dt)
            tokens = time_tokens + tokens
        return tokens

    @staticmethod
    def trim_tokens_back(
        duration: float,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        tokens: list[str],
    ):
        t = 0
        tokens = tokens.copy()
        result = []
        for token in tokens:
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
            result.append(token)
            if t >= duration:
                break
        return result

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        input_tokens = tokenizer.tokenize(prompt_notes)
        prompt_notes = tokenizer.untokenize(input_tokens)

        step_target_tokens = []
        output_tokens = []
        source_token = self.task_generator.source_token
        target_token = self.task_generator.target_token

        for _ in range(self.max_new_tokens):
            step_input_tokens = input_tokens[: self.prompt_context_length]

            source_token = self.task_generator.source_token
            target_token = self.task_generator.target_token

            step_input_tokens = [source_token] + step_input_tokens + [target_token] + step_target_tokens

            step_token_ids = [tokenizer.token_to_id[token] for token in step_input_tokens]

            step_token_ids = torch.tensor(
                [step_token_ids],
                device=device,
                dtype=torch.int64,
            )
            next_token = model.generate_new_tokens(
                idx=step_token_ids,
                temperature=self.temperature,
                max_new_tokens=1,
            )

            next_token_id = next_token[0].cpu().numpy()[0]
            next_token = tokenizer.vocab[next_token_id]

            output_tokens.append(next_token)
            step_target_tokens.append(next_token)

            generated_notes_length = len(step_target_tokens)
            # If the generated notes are longer than context_duration, move the generation window time_step to the right
            if generated_notes_length > self.target_context_length:
                input_tokens = SeqToSeqTokenwiseGenerator.trim_tokens_front(
                    time_step=self.time_step,
                    tokenizer=tokenizer,
                    tokens=input_tokens,
                )
                step_target_tokens = SeqToSeqTokenwiseGenerator.trim_tokens_front(
                    time_step=self.time_step,
                    tokenizer=tokenizer,
                    tokens=step_target_tokens,
                )

            if (
                SeqToSeqTokenwiseGenerator.calculate_token_duration(
                    tokenizer=tokenizer,
                    tokens=input_tokens,
                )
                == 0
            ):
                break
        target_notes = tokenizer.untokenize(tokens=output_tokens)
        return prompt_notes, target_notes


class NoteToNoteGenerator(MidiGenerator):
    """
    This generation method assures equal number of notes in prompt and target contexts at all times
    during generation. It is ideal for tasks such as velocity denoising or performance task -
    where there should be one note generated per one note in the prompt.
    """

    def __init__(
        self,
        task: str,
        prompt_context_notes: int,
        target_context_notes: int,
        step: int,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ):
        super().__init__(task=task)
        self.prompt_context_notes = prompt_context_notes
        self.target_context_notes = target_context_notes
        self.step = step
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.task_generator = Task.get_task(task_name=task)

    @staticmethod
    def default_parameters() -> dict:
        return {
            "prompt_context_notes": 128,
            "target_context_notes": 64,
            "step": 16,
            "max_new_tokens": 1024,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "prompt_context_length": self.prompt_context_notes,
            "target_context_length": self.target_context_notes,
            "time_step": self.time_step,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    @staticmethod
    def calculate_token_notes(
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        tokens: list[str],
    ):
        try:
            notes = tokenizer.untokenize(
                tokens=tokens,
                complete_notes=False,
            )
        except KeyError:
            # KeyError in tokenizer.untokenizes is raised when there are no full notes in tokens.
            return 0
        return len(notes)

    @staticmethod
    def trim_notes_front(
        step: int,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        tokens: list[str],
    ):
        notes = tokenizer.untokenize(
            tokens=tokens,
            complete_notes=False,
        )
        offset = notes.iloc[step:].start.min()
        notes = notes.iloc[step:]
        notes.start -= offset
        notes.end -= offset
        return tokenizer.tokenize(notes)

    @staticmethod
    def trim_notes_back(
        size: float,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        tokens: list[str],
    ):
        notes = tokenizer.untokenize(
            tokens=tokens,
            complete_notes=False,
        )
        notes = notes.iloc[:size]
        return tokenizer.tokenize(notes)

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        input_tokens = tokenizer.tokenize(prompt_notes)
        prompt_notes = tokenizer.untokenize(input_tokens)

        step_target_tokens = []
        output_tokens = []
        source_token = self.task_generator.source_token
        target_token = self.task_generator.target_token

        for _ in range(self.max_new_tokens):
            step_input_tokens = self.trim_notes_back(
                size=self.prompt_context_notes,
                tokenizer=tokenizer,
                tokens=input_tokens,
            )

            source_token = self.task_generator.source_token
            target_token = self.task_generator.target_token

            step_input_tokens = [source_token] + step_input_tokens + [target_token] + step_target_tokens

            step_token_ids = [tokenizer.token_to_id[token] for token in step_input_tokens]

            step_token_ids = torch.tensor(
                [step_token_ids],
                device=device,
                dtype=torch.int64,
            )
            next_token = model.generate_new_tokens(
                idx=step_token_ids,
                temperature=self.temperature,
                max_new_tokens=1,
            )

            next_token_id = next_token[0].cpu().numpy()[0]
            next_token = tokenizer.vocab[next_token_id]

            output_tokens.append(next_token)
            step_target_tokens.append(next_token)

            generated_notes_size = self.calculate_token_notes(
                tokenizer=tokenizer,
                tokens=step_target_tokens,
            )
            # If the generated notes are longer than context_duration, move the generation window time_step to the right
            if generated_notes_size > self.target_context_notes:
                st.write("STEP")
                st.write(
                    tokenizer.untokenize(
                        step_input_tokens,
                        complete_notes=False,
                    )
                )
                st.write(
                    tokenizer.untokenize(
                        step_target_tokens,
                        complete_notes=False,
                    )
                )
                input_tokens = NoteToNoteGenerator.trim_notes_front(
                    step=self.step,
                    tokenizer=tokenizer,
                    tokens=input_tokens,
                )
                step_target_tokens = NoteToNoteGenerator.trim_notes_front(
                    step=self.step,
                    tokenizer=tokenizer,
                    tokens=step_target_tokens,
                )
            if (
                NoteToNoteGenerator.calculate_token_notes(
                    tokenizer=tokenizer,
                    tokens=input_tokens,
                )
                == 0
            ):
                break
        target_notes = tokenizer.untokenize(
            tokens=output_tokens,
            complete_notes=False,
        )
        return prompt_notes, target_notes


generator_types = {
    "NextTokenGenerator": NextTokenGenerator,
    "NextTokenTokenwiseGenerator": NextTokenTokenwiseGenerator,
    "SeqToSeqIterativeGenerator": SeqToSeqIterativeGenerator,
    "SeqToSeqTokenwiseGenerator": SeqToSeqTokenwiseGenerator,
    "NoteToNoteGenerator": NoteToNoteGenerator,
}
