import pandas as pd
from pandas import DataFrame
from tokenizers import Tokenizer
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers import MidiTokenizer, ExponentialTimeTokenizer


class ExponentialTokenizer(ExponentialTimeTokenizer):
    def __init__(
        self,
        min_time_unit: float = 0.01,
        n_velocity_bins: int = 128,
        special_tokens: list[str] = None,
    ):
        super().__init__(min_time_unit, n_velocity_bins, special_tokens)
        self.pad_token_id = self.token_to_id["<PAD>"]

    def encode(
        self,
        notes: DataFrame,
        pad_to_size: int = 0,
        prefix_tokens: list[str] = [],
    ) -> list[int]:
        encoding = super().encode(notes)

        padding_size = pad_to_size - len(encoding) - len(prefix_tokens)

        suffix_ids = [self.token_to_id[token] for token in prefix_tokens]
        padding = [self.pad_token_id] * padding_size

        return suffix_ids + encoding + padding

    def tokenize(self, notes: pd.DataFrame) -> list[str]:
        notes = self.quantize_frame(notes)
        notes = notes.sort_values(by="pitch", kind="stable")
        notes.sort_values(by="start", kind="stable")
        events = self._notes_to_events(notes)

        tokens = []
        previous_time = 0

        for event in events:
            dt = event["time"] - previous_time
            tokens.extend(self.tokenize_time_distance(dt))
            tokens.append(self.velocity_bin_to_token[event["velocity_bin"]])
            if event["event"] == "NOTE_ON":
                tokens.append(self.pitch_to_on_token[event["pitch"]])
            else:
                tokens.append(self.pitch_to_off_token[event["pitch"]])
            previous_time = event["time"]

        return tokens

    def untokenize(self, tokens: list[str], complete_notes: bool = False) -> pd.DataFrame:
        events = []
        current_time = 0
        current_velocity = 0

        for token in tokens:
            if token.endswith("T"):
                current_time += self.token_to_dt[token]
            elif token.startswith("VELOCITY"):
                current_velocity = self.bin_to_velocity[self.token_to_velocity_bin[token]]
            elif token.startswith("NOTE_ON"):
                events.append((current_time, "on", self.token_to_pitch[token], current_velocity))
            elif token.startswith("NOTE_OFF"):
                events.append((current_time, "off", self.token_to_pitch[token], 0))

        events.sort()  # Sort by time
        notes = []
        open_notes = {}

        for time, event_type, pitch, velocity in events:
            if event_type == "on":
                if pitch in open_notes:
                    # Close the previous note if it's still open
                    start, vel = open_notes.pop(pitch)
                    notes.append({"pitch": pitch, "start": start, "end": time, "velocity": vel})
                open_notes[pitch] = (time, velocity)
            elif event_type == "off":
                if pitch in open_notes:
                    start, vel = open_notes.pop(pitch)
                    notes.append({"pitch": pitch, "start": start, "end": time, "velocity": vel})

        # Close any remaining open notes
        if complete_notes:
            for pitch, (start, vel) in open_notes.items():
                notes.append({"pitch": pitch, "start": start, "end": time, "velocity": vel})

        notes_df = pd.DataFrame(notes)
        if not notes_df.empty:
            notes_df.loc[notes_df["end"] == notes_df["start"], "end"] += self.min_time_unit
            notes_df = notes_df.sort_values(by="pitch", kind="stable")
            notes_df = notes_df.sort_values("start", kind="stable").reset_index(drop=True)
        return notes_df

    def _notes_to_events(self, notes: pd.DataFrame) -> list[dict]:
        events = []
        for _, note in notes.iterrows():
            events.append(
                {
                    "time": note["start"],
                    "event": "NOTE_ON",
                    "pitch": note["pitch"],
                    "velocity_bin": note["velocity_bin"],
                }
            )
            events.append(
                {
                    "time": note["end"],
                    "event": "NOTE_OFF",
                    "pitch": note["pitch"],
                    "velocity_bin": 0,
                }
            )
        return sorted(events, key=lambda x: (x["time"], x["event"] != "NOTE_OFF"))


class AwesomeTokenizer(AwesomeMidiTokenizer):
    def __init__(
        self,
        base_tokenizer: MidiTokenizer,
        bpe_tokenizer: Tokenizer = None,
        max_vocab_size: int = 30000,
        max_token_length: int = 128,
        special_tokens: list[str] = None,
    ):
        super().__init__(
            base_tokenizer,
            bpe_tokenizer,
            max_vocab_size,
            max_token_length,
            special_tokens,
        )
        self.pad_token_id = self.token_to_id["<PAD>"]

    def encode(
        self,
        notes: DataFrame,
        pad_to_size: int = 0,
        prefix_tokens: list[str] = [],
    ) -> list[int]:
        encoding = super().encode(notes)
        padding_size = pad_to_size - len(encoding) - len(prefix_tokens)
        suffix_ids = [self.token_to_id[token] for token in prefix_tokens]
        padding = [self.pad_token_id] * padding_size

        return suffix_ids + encoding + padding
