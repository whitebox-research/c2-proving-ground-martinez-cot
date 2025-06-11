#!/usr/bin/env python3

import logging
from pathlib import Path
import click

from src.cot_splitting import split_cot_responses
from src.typing import *
from src.utils import setup_logging


@click.command()
@click.argument("responses_path", type=click.Path(exists=True))
@click.option(
    "--model_id",
    "-s",
    type=str,
    default="anthropic/claude-3.5-sonnet",
    help="Model ID to use for splitting CoT responses using Anthropic API. "
    "For Claude models, use format like 'anthropic/claude-3.5-sonnet'.",
)
@click.option(
    "--max_retries",
    "-r",
    type=int,
    default=1,
    help="Maximum retries for splitting CoT responses with the each model",
)
@click.option(
    "--max_parallel",
    "-p",
    type=int,
    default=None,
    help="Maximum number of parallel requests. If not set, it will use the Anthropic API limits.",
)
@click.option(
    "--max_new_tokens_override",
    "-t",
    type=int,
    default=None,
    help="Override the max_new_tokens parameter for the model. If not set, will use 1.25x the original max_new_tokens.",
)
@click.option(
    "--prefix",
    type=int,
    default=None,
    help="Only process the first N items in the batch. If not set, process all items.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Increase verbosity (can be used multiple times)",
)
def main(
    responses_path: str,
    model_id: str,
    max_retries: int,
    verbose: int,
    max_parallel: int | None,
    max_new_tokens_override: int | None,
    prefix: int | None,
):
    # Set up logging to both console and file
    log_path = setup_logging(verbose, "putnamlike2_split_cots")
    
    logging.warning("WARNING! This is somewhat unreliable, particularly for really long rollouts, as it does only very basic checks of the correct format by checking that the length of the steps added together is within 10% of the original response length")
    cot_responses = CotResponses.load(Path(responses_path))
    results = split_cot_responses(
        responses=cot_responses,
        model_id=model_id,
        max_retries=max_retries,
        max_parallel=max_parallel,
        max_new_tokens_override=max_new_tokens_override,
        prefix=prefix,
    )
    path = responses_path
    suffix = "_splitted"
    path = str(path)
    # Change blah/blah2.txt -> blah/blah2_suffix.txt
    path_split = path.split(".")
    path_split[-2] = path_split[-2] + suffix
    path = Path(".".join(path_split))

    path = results.save(path=path)
    logging.error(f"Saved split CoT responses to {path}")


if __name__ == "__main__":
    main()
