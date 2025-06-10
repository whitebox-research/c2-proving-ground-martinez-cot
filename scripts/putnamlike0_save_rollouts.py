#!/usr/bin/env python3

import asyncio
import logging
import os
import uuid
import datetime
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
import yaml

from src.api_utils.anthropic_utils import ANBatchProcessor, ANRateLimiter
from src.typing import (
    CotResponses,
    DefaultSamplingParams,
    MathDatasetParams,
    MathQsDataset,
    MathQuestion,
    MathResponse,
)


def setup_logging(verbose: bool, script_name: str) -> str:
    """Set up logging to both console and file.
    
    Args:
        verbose: Whether to log at INFO level (True) or WARNING level (False)
        script_name: Name of the script for the log filename
    
    Returns:
        Path to the log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_path = logs_dir / log_filename
    
    # Set up logging
    log_level = logging.INFO if verbose else logging.WARNING
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging to {log_path}")
    return str(log_path)


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
                image_path=f"putnam_problems_images/{row['problem_name']}_stmt.png"
            )
            for _, row in df.iterrows()
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
    force_open_router: bool = False,
    track_api_usage: str = "none",
):
    """Create the appropriate processor based on the model ID."""

    def get_tuple_or_str_response(
        response: tuple[str, str] | str, other: Any
    ) -> tuple[str | None, str]:
        logging.info(f"Inner response: {response}")

        if isinstance(response, tuple):
            assert (
                len(response) == 2
            ), f"Expected tuple of length 2, got {len(response)}"
            return response
        else:
            return (None, response)

    if "anthropic" in model_id:
        # OpenRouter processor
        logging.info(f"Using Anthropic model {model_id}")
        an_rate_limiter = None
        if max_parallel is not None:
            an_rate_limiter = ANRateLimiter(
                requests_per_interval=max_parallel,
                tokens_per_interval=10000,
                interval_seconds=480,
            )

            processor = ANBatchProcessor[MathResponse, MathResponse](
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
    force_open_router: bool = False,
    preamble: str = "",
    track_api_usage: str = "none",
) -> CotResponses:
    """Generate rollouts for each problem in the dataset."""

    processor = create_processor(
        model_id=model_id,
        max_retries=max_retries,
        max_parallel=max_parallel,
        force_open_router=force_open_router,
        track_api_usage=track_api_usage
    )

    # Prepare questions for processing
    questions = dataset.questions[:prefix] if prefix else dataset.questions

    logging.warning("USING A THINK STEP-BY-STEP PREFIX!")
    batch_items = [
        (
            q,
            f"{preamble}{q.problem}",
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
                solution=question.solution,
                image_path=question.image_path,
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
    "-s",
    type=str,
    default="anthropic/claude-3-opus",
    help="Model ID for generating rollouts (OpenRouter or DeepSeek model)",
)
@click.option(
    "--max_retries",
    "-r",
    type=int,
    default=1,
    help="Maximum retries for failed requests",
)
@click.option(
    "--max_parallel",
    "-p",
    type=int,
    default=None,
    help="Maximum number of parallel requests",
)
@click.option(
    "--prefix",
    "-prefix",
    type=int,
    default=None,
    help="Only process the first N problems",
)
@click.option(
    "--preamble",
    type=str,
    default="Solve this math problem step-by-step, reasoning first and then producing an answer.\n\n",
    help="Preamble text to add before each problem",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--open_router",
    is_flag=True,
    help="Force using OpenRouter even for DeepSeek models",
)
def main(
    input_yaml: str,
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    prefix: Optional[int],
    verbose: bool,
    open_router: bool,
    preamble: str,
):
    """Generate rollouts for Putnam problems using OpenRouter or DeepSeek models."""
    # Set up logging to both console and file
    log_path = setup_logging(verbose, "putnamlike0_texts_save_rollouts")
    logging.info(f"Loading dataset from {input_yaml}")

    # Load and prepare dataset
    input_path = Path(input_yaml)
    df = load_putnam_results_as_df(input_path)
    dataset = create_putnam_dataset(df)

    # Generate rollouts
    results = asyncio.run(
        generate_rollouts(
            dataset=dataset,
            model_id=model_id,
            preamble=preamble,
            max_retries=max_retries,
            max_parallel=max_parallel,
            prefix=prefix,
            force_open_router=open_router,
        )
    )

    # Save results
    for i in range(0, 100):
        output_path = results.get_path(
            f"_texts_v{i}" + (f"_prefix_{prefix}" if prefix else "")
        )
        if not os.path.exists(output_path):
            break

    saved_path = results.save(path=output_path)
    logging.info(f"Saved rollouts to {saved_path}")


if __name__ == "__main__":
    main()
