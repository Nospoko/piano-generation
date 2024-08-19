import re
from abc import ABC, abstractmethod

import torch
import pandas as pd
from torch import nn

from generation.tasks import Task
from model.tokenizers import AwesomeTokenizer, ExponentialTokenizer


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
        {
            "task": "next_token_prediction",
        }


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


class SeqToSeqTokenwiseGenerator(MidiGenerator):
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


class SeqToSeqIterativeGenerator(MidiGenerator):
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

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.generate_subsequence_iteratively(
            prompt_notes=prompt_notes,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    def generate_subsequence_iteratively(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
    ) -> pd.DataFrame:
        # Tokenize prompt at the beginning to standarize tokenization during generation.
        prompt_notes = tokenizer.untokenize(tokenizer.tokenize(prompt_notes))
        # Initialize the first step with notes within the prompt and target context durations
        step_prompt_notes = prompt_notes[prompt_notes.end < self.prompt_context_duration].copy()
        step_target_notes = pd.DataFrame(columns=prompt_notes.columns)

        # Initialize the list of all target notes with the initial target notes
        all_target_notes = [step_target_notes]
        time = 0
        end = prompt_notes.end.max()
        it = 0
        # Iterate through the piece, generating bass notes in steps
        while time <= end:
            # Calculate the start offset for the bass notes in this step
            start_offset = it * self.time_step
            it += 1
            step_prompt_notes.start -= start_offset
            step_prompt_notes.end -= start_offset

            step_target_notes = step_target_notes[(step_target_notes.start > 0) & (step_target_notes.end > 0)]

            # Tokenize the current step's prompt and target notes
            step_sequence = tokenizer.tokenize(step_prompt_notes)
            step_target = tokenizer.tokenize(step_target_notes)

            # Combine prompt, bass marker, and target into input sequence
            source_task_token = self.task_generator.source_token
            target_task_token = self.task_generator.target_token

            input_sequence = [source_task_token] + step_sequence + [target_task_token] + step_target
            # print(input_sequence)
            # Convert tokens to ids and prepare input tensor
            input_token_ids = torch.tensor(
                [[tokenizer.token_to_id[token] for token in input_sequence]],
                device=device,
                dtype=torch.int64,
            )
            # Generate new tokens using the model
            output = model.generate_new_tokens(
                idx=input_token_ids,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            # Convert output to numpy array and decode tokens
            output = output[0].cpu().numpy()

            # Convert target tokens back to notes
            output_target_notes = tokenizer.decode(output)

            # Select only the newly generated notes within the current time step
            notes_within_step = output_target_notes.end < self.target_context_duration + self.time_step
            target_notes = output_target_notes[notes_within_step].copy()
            step_target_notes = target_notes.copy()
            # FIXME: This does not include the target notes generated in the previous step.
            print(step_target_notes)

            # Adjust the start and end times of the bass notes
            target_notes.start += start_offset
            target_notes.end += start_offset
            target_notes["duration"] = target_notes.end - target_notes.start

            # Add the generated bass notes to the collection
            all_target_notes.append(target_notes)
            # Prepare for the next iteration:
            # Select the prompt notes for the next time step
            time = time + self.time_step
            prompt_selector = (prompt_notes.start > time) & (prompt_notes.end < time + self.prompt_context_duration)
            step_prompt_notes = prompt_notes[prompt_selector].copy()
            step_target_notes = step_target_notes[step_target_notes.start > self.time_step]
            step_target_notes.start -= self.time_step
            step_target_notes.end -= self.time_step

        # Combine all generated bass notes and return
        target_notes = pd.concat(all_target_notes[1:]).reset_index(drop=True)
        return prompt_notes, target_notes


generator_types = {
    "NextTokenGenerator": NextTokenGenerator,
    "SeqToSeqIterativeGenerator": SeqToSeqIterativeGenerator,
    "SeqToSeqTokenwiseGenerator": SeqToSeqTokenwiseGenerator,
}
