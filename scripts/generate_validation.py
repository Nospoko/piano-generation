import os
import argparse
from typing import List, Union
from contextlib import nullcontext

import torch
import pandas as pd

import piano_generation.generation.generators as generators
import piano_generation.database.database_manager as database_manager
from piano_generation.utils import load_cfg, load_tokenizer, initialize_gpt_model
from piano_generation import GPT, Task, AwesomeTokenizer, ExponentialTokenizer, task_map


def run_generation_step(
    model: GPT,
    checkpoint: dict,
    run_name: str,
    validation_examples: pd.DataFrame,
    task: Task,
    generator: generators.MidiGenerator,
    tokenizer: Union[AwesomeTokenizer, ExponentialTokenizer],
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
            generator=generator,
            generated_notes=generation,
            prompt_notes=prompt_notes,
            source_notes=source_notes,
            source=source,
        )


def run_validation_for_task(
    model: GPT,
    checkpoint: dict,
    run_name: str,
    validation_examples: pd.DataFrame,
    task_name: str,
    tokenizer: Union[AwesomeTokenizer, ExponentialTokenizer],
    device: torch.device,
    ctx: nullcontext,
):
    task = Task.get_task(task_name=task_name)
    generator = generators.SeqToSeqTokenwiseGenerator(
        task=task_name,
        prompt_context_length=1024,
        target_context_length=512,
        time_step=2,
        temperature=1.0,
        max_new_tokens=4096,
    )

    print(f"Running validation for task: {task_name}")
    with ctx:
        run_generation_step(
            model=model,
            checkpoint=checkpoint,
            run_name=f"{run_name}_{task_name}",
            validation_examples=validation_examples,
            tokenizer=tokenizer,
            generator=generator,
            task=task,
            device=device,
        )


def main(model_path: str, device: str, tasks: List[str]):
    checkpoint = torch.load(f=model_path, map_location=device)

    cfg = load_cfg(checkpoint=checkpoint)
    tokenizer = load_tokenizer(cfg=cfg)

    model = initialize_gpt_model(
        cfg=cfg,
        checkpoint=checkpoint,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    validation_examples = database_manager.get_validation_sources()

    run_name = os.path.splitext(os.path.basename(model_path))[0]

    for task_name in tasks:
        run_validation_for_task(
            model=model,
            checkpoint=checkpoint,
            run_name=run_name,
            validation_examples=validation_examples,
            task_name=task_name,
            tokenizer=tokenizer,
            device=device,
            ctx=ctx,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model generation on validation examples for multiple tasks.")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("device", type=str, help="Device to perform calculations on.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(task_map.keys()),
        help="List of tasks to run validation on. If not specified, all tasks will be run.",
    )
    args = parser.parse_args()

    main(model_path=args.model_path, device=args.device, tasks=args.tasks)
