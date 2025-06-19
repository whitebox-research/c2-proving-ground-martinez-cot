#!/usr/bin/env python3

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
import yaml

from src.anthropic_utils import ANBatchProcessorWithImage, ANRateLimiter
from src.google_utils import GOBatchProcessor, GORateLimiter
from src.typing import (
    CotResponses,
    DefaultSamplingParams,
    MathDatasetParams,
    MathQsDataset,
    MathQuestion,
    MathResponse,
)
from src.utils import setup_logging


def load_putnam_results_as_df(yaml_path: Path) -> pd.DataFrame:
    """Load Putnam results from YAML into a pandas DataFrame."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return pd.DataFrame(data)


def create_putnam_dataset(df: pd.DataFrame) -> MathQsDataset:
    """Create a MathQsDataset from a Putnam DataFrame."""
    # Sort problems by year and type
    df = df.sort_values(
        by="problem_name",
        key=lambda x: pd.Series(
            [
                # Extract year and problem type (e.g. 'a1', 'b2')
                (int(name.split("_")[1]), name.split("_")[2])
                for name in x
            ]
        ).map(
            lambda t: (
                {
                    "a1": 0,
                    "b1": 1,
                    "a2": 2,
                    "b2": 3,
                    "a3": 4,
                    "b3": 5,
                    "a4": 6,
                    "b4": 7,
                    "a5": 8,
                    "b5": 9,
                    "a6": 10,
                    "b6": 11,
                }[t[1]],
                -t[0],
            )
        ),
    )

    return MathQsDataset(
        questions=[
            MathQuestion(
                name=row["problem_name"],
                problem=row["informal_statement"],
                solution=row["informal_solution"],
                image_path=f"data/dataset/putnam_problems_images/{row['problem_name']}_stmt.png", #TODO: Add model usage tokens and API response time
            )
            for _, row in df.iterrows() if os.path.exists(f"data/dataset/putnam_problems_images/{row['problem_name']}_stmt.png")
        ],
        params=MathDatasetParams(
            description="Putnam Competition Problems",
            id="filtered_putnambench",
            pre_id=None,
        ),
    )


def create_processor(
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    is_text: bool = False
):
    """Create the appropriate processor based on the model ID."""

    def get_tuple_or_str_response(
        response: tuple[str, str] | str, other: Any
    ) -> tuple[str | None, str]:

        if isinstance(response, tuple):
            assert (
                len(response) == 2
            ), f"Expected tuple of length 2, got {len(response)}"
            return response
        else:
            return (None, response)

    if "gemini" in model_id:
        # Google processor
        logging.info(f"Using model {model_id}")
        go_rate_limiter = None
        if max_parallel is not None:
            go_rate_limiter = GORateLimiter(
                requests_per_interval=max_parallel,
                tokens_per_interval=100000,
                interval_seconds=1,
            )

        processor = GOBatchProcessor[MathResponse, MathResponse](
            model_id=model_id,
            max_retries=max_retries,
            max_new_tokens=1000,
            temperature=0.0,
            process_response=get_tuple_or_str_response,
            rate_limiter=go_rate_limiter,
            is_text=is_text,  # Use text-only if specified
        )

        return processor
    else:
        logging.info(f"Using Anthropic model {model_id}")
        an_rate_limiter = None
        if max_parallel is not None:
            an_rate_limiter = ANRateLimiter(
                requests_per_interval=max_parallel,
                tokens_per_interval=100000,
                interval_seconds=60,
            )

            processor = ANBatchProcessorWithImage[MathResponse, MathResponse](
                model_id=model_id,
                max_retries=max_retries,
                max_new_tokens=1000,
                temperature=0.0,
                process_response=get_tuple_or_str_response,
                rate_limiter=an_rate_limiter,
                # track_api_usage=track_api_usage,
            )
        return processor


async def generate_rollouts(
    dataset: MathQsDataset,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    prefix: Optional[int] = None,
    preamble: str = "",
    is_text: bool = False
) -> CotResponses:
    """Generate rollouts for each problem in the dataset."""

    processor = create_processor(
        model_id=model_id,
        max_retries=max_retries,
        max_parallel=max_parallel,
        is_text=is_text,
    )

    # Prepare questions for processing
    questions = dataset.questions[:prefix] if prefix else dataset.questions

    logging.warning("USING A THINK STEP-BY-STEP PREFIX!")
    batch_items = [
        (
            q,
            f"{preamble}",
        )
        for q in questions
    ]  # TODO(arthur): Fully verify we used this format for all

    # Process all questions in batch
    logging.info(f"Processing {len(batch_items)} problems")
    results = await processor.process_batch(batch_items)

    # Collect responses
    responses_by_qid = {}
    for question, (_, thinking_and_answer) in zip(questions, results):
        if thinking_and_answer is None or thinking_and_answer[-1] is None:
            logging.warning(
                f"Skipping failed response for {question.name} {thinking_and_answer=}"
            )
            continue

        thinking, answer = thinking_and_answer

        responses_by_qid[question.name] = {
            str(uuid.uuid4())[:8]: MathResponse(
                name=question.name,
                problem=question.problem,
                image_path=question.image_path,
                solution=question.solution,
                model_thinking=thinking,
                model_answer=[answer],  # Unsplit
            )
        }

    return CotResponses(
        responses_by_qid=responses_by_qid,
        model_id=model_id,
        instr_id="instr-v0",
        ds_params=dataset.params,
        sampling_params=DefaultSamplingParams(),
    )

@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.option(
    "--model_id",
    type=str,
    default="gemini-2.0-flash-thinking-",
    help="Model for generating rollouts",
)
@click.option(
    "--max_retries",
    type=int,
    default=1,
    help="Maximum retries for failed requests",
)
@click.option(
    "--max_parallel",
    type=int,
    default=1,
    help="Maximum number of parallel requests",
)
@click.option(
    "--prefix",
    type=int,
    default=None,
    help="Only process N problems",
)
@click.option(
    "--preamble",
    type=str,
    default="Solve this math problem step-by-step, reasoning first and then producing an answer.\n\n",
    help="Preamble text to add before each problem",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--text",
    is_flag=True,
    help="Use text inputs (default is to use image inputs)",
)
def main(
    input_yaml: str,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    prefix: Optional[int],
    verbose: bool,
    preamble: str,
    text:bool = False,
):
    """Generate rollouts for Putnam problems"""
    # Set up logging to both console and file
    log_path = setup_logging(verbose, "putnamlike0_save_rollouts")
    logging.info(f"Loading dataset from {input_yaml}")

    # Load and prepare dataset
    input_path = Path(input_yaml)
    df = load_putnam_results_as_df(input_path)
    dataset = create_putnam_dataset(df)

    logging.info(f"Running prompt:\n {preamble}")
    # Generate rollouts
    results = asyncio.run(
        generate_rollouts(
            dataset=dataset,
            model_id=model_id,
            preamble=preamble,
            max_retries=max_retries,
            max_parallel=max_parallel,
            prefix=prefix,
            is_text=text,  # Use text-only if specified
        )
    )

    # Save results
    for i in range(0, 100):
        if text: output_path = results.get_path(f"_text_v{i}" + (f"_prefix_{prefix}" if prefix else ""))
        else: output_path = results.get_path(f"_images_v{i}" + (f"_prefix_{prefix}" if prefix else ""))
        
        if not os.path.exists(output_path):
            break

    saved_path = results.save(path=output_path)
    logging.info(f"Saved rollouts to {saved_path}")


if __name__ == "__main__":
    main()
