from .static_generator import StaticGenerator
from .static_bpe_generator import StaticBpeGenerator
from .note_to_note_generator import NoteToNoteGenerator
from .next_token_generator import NextTokenGenerator, NextTokenTokenwiseGenerator
from .seq_to_seq_generator import SeqToSeqIterativeGenerator, SeqToSeqTokenwiseGenerator

generator_types = {
    "NextTokenGenerator": NextTokenGenerator,
    "NextTokenTokenwiseGenerator": NextTokenTokenwiseGenerator,
    "SeqToSeqIterativeGenerator": SeqToSeqIterativeGenerator,
    "SeqToSeqTokenwiseGenerator": SeqToSeqTokenwiseGenerator,
    "NoteToNoteGenerator": NoteToNoteGenerator,
    "StaticGenerator": StaticGenerator,
    "StaticBpeGenerator": StaticBpeGenerator,
}
