# Piano Generation

Piano Generation is a Python package for generating piano music using machine learning models. It provides tools for training models on MIDI data and generating new piano compositions.

## Features

- MIDI data processing and tokenization
- Various generator types (NextToken, SeqToSeq, etc.)
- Support for different tasks (e.g., next token prediction, denoising)
- Integration with PyTorch for model training and inference
- Utility functions for model management and data handling
- Database integration for storing and managing generated music

## Installation

To install the Piano Generation package, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/piano-generation.git
   cd piano-generation
   ```

2. Install the package in editable mode:
   ```
   pip install -e .
   ```

## Usage

Here's a basic example of how to use the Piano Generation package:

```sh
python -m scripts.download_model model_to_download.pt
```

```python
from piano_generation import GPT, Task, MidiGenerator
from midi_tokenizers import ExponentialTimeTokenizer
from piano_generation.utils import load_cfg, load_tokenizer, load_checkpoint, initialize_gpt_model
from piano_generation.database import database_manager

# Load a pre-trained model
checkpoint = load_checkpoint("checkpoints/downloaded_checkpoint.pt")
cfg = load_cfg(checkpoint)
tokenizer = load_tokenizer(cfg)
model = initialize_gpt_model(cfg, checkpoint, device="cuda")

# Create a generator
generator = MidiGenerator.get_generator(
    generator_name="SeqToSeqTokenwiseGenerator",
    parameters={
        "task": "next_token_prediction",
        "prompt_context_length": 1024,
        "target_context_length": 512,
        "time_step": 2,
        "temperature": 1.0,
        "max_new_tokens": 4096,
    }
)

# Generate music
prompt_notes, generated_notes = generator.generate(
    prompt_notes=your_input_notes,
    model=model,
    tokenizer=tokenizer,
    device="cuda"
)

# Store the generated music in the database
database_manager.insert_generation(
    model_checkpoint=checkpoint,
    model_name="your_model_name",
    generator=generator,
    generated_notes=generated_notes,
    prompt_notes=prompt_notes,
    source_notes=your_input_notes,
    source={"source_info": "your_source_info"}
)

# Process the generated notes as needed
```

## Project Structure

```
piano_generation/
├── database/
│   ├── database_connection.py
│   └── database_manager.py
├── generation/
│   ├── generators.py
│   └── tasks.py
├── model/
│   ├── dummy.py
│   ├── gpt2.py
│   └── tokenizers.py
└── utils.py
```
