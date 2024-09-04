from .note_to_note_generator import NoteToNoteGenerator
from .next_token_generator import NextTokenGenerator, NextTokenTokenwiseGenerator
from .seq_to_seq_generator import SeqToSeqIterativeGenerator, SeqToSeqTokenwiseGenerator

generator_types = {
    "NextTokenGenerator": NextTokenGenerator,
    "NextTokenTokenwiseGenerator": NextTokenTokenwiseGenerator,
    "SeqToSeqIterativeGenerator": SeqToSeqIterativeGenerator,
    "SeqToSeqTokenwiseGenerator": SeqToSeqTokenwiseGenerator,
    "NoteToNoteGenerator": NoteToNoteGenerator,
}
