from abc import ABC, abstractmethod

import torch
import pandas as pd
from torch import nn
from tasks import Task

from model.tokenizers import AwesomeTokenizer, ExponentialTokenizer


class MidiGenerator(ABC):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def generate(
        self,
        prompt_notes: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass


class NextTokenGenerator(MidiGenerator):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
        prompt_context_duration: float = 15.0,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        self.prompt_context_duration = prompt_context_duration
        max_new_tokens = max_new_tokens
        temperature = temperature

    def generate(
        self,
        prompt_notes: pd.DataFrame,
    ):
        prompt_notes = prompt_notes[prompt_notes.end < self.prompt_context_duration]

        # Tokenize prompt notes
        input_sequence = self.tokenizer.tokenize(prompt_notes)

        # Convert tokens to ids and prepare input tensor
        input_token_ids = torch.tensor(
            [[self.tokenizer.token_to_id[token] for token in input_sequence]],
            device=self.device,
            dtype=torch.int64,
        )

        # Generate new tokens using the model
        with torch.no_grad():
            output = self.model.generate_new_tokens(
                input_token_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        # Convert output to numpy array and decode tokens
        output = output[0].cpu().numpy()
        out_tokens = [self.tokenizer.vocab[token_id] for token_id in output]

        # Convert tokens back to notes
        generated_notes = self.tokenizer.untokenize(out_tokens)

        return prompt_notes, generated_notes


class SeqToSeqIterativeGenerator(MidiGenerator):
    def __init__(
        self,
        task: str,
        model: nn.Module,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        device: torch.device,
        prompt_context_duration: float,
        target_context_duration: float,
        time_step: float,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        self.prompt_context_duration = prompt_context_duration
        self.target_context_duration = target_context_duration
        self.time_step = time_step
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.task_generator = Task.get_task(task_name=task)

    def prepare_prompt(notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # TODO: prompt preparation
        pass

    def generate(self, prompt_notes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        prompt_notes, target_notes = self.prepare_prompt(notes=prompt_notes)
        return self.generate_subsequence_iteratively(prompt_notes=prompt_notes, target_notes=target_notes)

    def generate_subsequence_iteratively(
        self,
        prompt_notes: pd.DataFrame,
        target_notes: pd.DataFrame,
    ) -> pd.DataFrame:
        # Tokenize prompt at the beginning to standarize tokenization during generation.
        prompt_notes = self.tokenizer.untokenize(self.tokenizer.tokenize(prompt_notes))
        # Initialize the first step with notes within the prompt and target context durations
        step_prompt_notes = prompt_notes[prompt_notes.end < self.prompt_context_duration].copy()
        step_target_notes = target_notes[target_notes.end < self.target_context_duration].copy()
        # Initialize the list of all target notes with the initial target notes
        all_target_notes = [step_target_notes]
        time = 0
        end = prompt_notes.end.max()

        # Handle the case where there's no target context
        if self.target_context_duration == 0:
            step_target_notes = pd.DataFrame(columns=prompt_notes.columns)
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
            step_sequence = self.tokenizer.tokenize(step_prompt_notes)
            step_target = self.tokenizer.tokenize(step_target_notes)

            # Combine prompt, bass marker, and target into input sequence
            source_task_token = self.task_generator.source_token
            target_task_token = self.task_generator.target_token

            input_sequence = [source_task_token] + step_sequence + [target_task_token] + step_target
            # Convert tokens to ids and prepare input tensor
            input_token_ids = torch.tensor(
                [[self.tokenizer.token_to_id[token] for token in input_sequence]],
                device=self.device,
                dtype=torch.int64,
            )
            # Generate new tokens using the model
            output = self.model.generate_new_tokens(
                idx=input_token_ids,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            # Convert output to numpy array and decode tokens
            output = output[0].cpu().numpy()
            out_tokens = [self.tokenizer.vocab[token_id] for token_id in output]

            # Extract target tokens (everything after the target notes marker)
            predict_command_position = out_tokens.index(target_task_token)
            target_tokens = out_tokens[predict_command_position:].copy()

            # Convert target tokens back to notes
            output_target_notes = self.tokenizer.untokenize(target_tokens)

            # Select only the newly generated notes within the current time step
            notes_within_step = output_target_notes.end < self.target_context_duration + self.time_step
            target_notes = output_target_notes[notes_within_step].copy()
            step_target_notes = target_notes.copy()

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
