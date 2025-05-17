#!/usr/bin/env python3

"""Example of using process_math_response for a real use case.

Example usage:
python -m dotenv run python scripts/putnam/math_response_example.py \
    --input_yaml path/to/input.yaml \
    --model_id "anthropic/claude-3.5-sonnet" \
    --verbose
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import click
import yaml

from chainscope.typing import (
    CotResponses,
    DefaultSamplingParams,
    MathDatasetParams,
    MathResponse,
)
from scripts.putnam.process_single_math_response import process_math_response


def load_math_response(yaml_path: Path, qid: Optional[str] = None) -> MathResponse:
    """Load a MathResponse from a YAML file.

    Args:
        yaml_path: Path to the YAML file
        qid: Optional question ID to load. If None, loads the first response found.

    Returns:
        MathResponse object
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    responses_by_qid = data.get("responses_by_qid", {})

    if not responses_by_qid:
        raise ValueError(f"No responses found in {yaml_path}")

    if qid is not None:
        # Get the specified response
        if qid not in responses_by_qid:
            raise ValueError(f"Question ID {qid} not found in {yaml_path}")

        # Get the first response for this qid
        response_data = next(iter(responses_by_qid[qid].values()))
    else:
        # Get the first response from the first qid
        first_qid = next(iter(responses_by_qid.keys()))
        response_data = next(iter(responses_by_qid[first_qid].values()))

    # Check if it's already a MathResponse or just a dictionary
    if isinstance(response_data, MathResponse):
        return response_data
    else:
        return MathResponse(
            name=response_data.get("name", "unknown"),
            problem=response_data.get("problem", ""),
            solution=response_data.get("solution", ""),
            model_answer=response_data.get("model_answer", []),
            model_thinking=response_data.get("model_thinking"),
            correctness_explanation=response_data.get("correctness_explanation"),
            correctness_is_correct=response_data.get("correctness_is_correct"),
            correctness_classification=response_data.get("correctness_classification"),
        )


def save_result(
    math_response: MathResponse,
    output_path: Path,
    model_id: str,
) -> None:
    """Save processed MathResponse to a YAML file.

    Args:
        math_response: Processed MathResponse
        output_path: Path to save the output
        model_id: Model ID used for processing
    """
    # Create a CotResponses object
    responses = {math_response.name: {"response_uuid": math_response}}

    # Create dataset params
    ds_params = MathDatasetParams(
        description=f"Processed Math Response with {model_id}",
        id="processed_math_response",
        pre_id=None,
    )

    # Create CotResponses object
    cot_responses = CotResponses(
        responses_by_qid=responses,
        model_id=model_id,
        instr_id="evaluation",
        ds_params=ds_params,
        sampling_params=DefaultSamplingParams(),
    )

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cot_responses.to_yaml_file(output_path)
    logging.info(f"Saved result to {output_path}")


@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.option(
    "--model_id",
    "-s",
    type=str,
    default="anthropic/claude-3.5-sonnet",
    help="Model for evaluation",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(),
    default=None,
    help="Path to save the processed result",
)
@click.option(
    "--qid",
    "-q",
    type=str,
    default=None,
    help="Specific question ID to process",
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
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def main(
    input_yaml: str,
    model_id: str,
    output_path: Optional[str],
    qid: Optional[str],
    max_retries: int,
    max_parallel: Optional[int],
    verbose: bool,
):
    """Process a MathResponse from a YAML file and save the result."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    input_path = Path(input_yaml)

    # Determine output path
    output_path_obj: Path
    if output_path is None:
        output_path_obj = input_path.parent / f"{input_path.stem}_processed.yaml"
    else:
        output_path_obj = Path(output_path)

    # Load math response
    math_response = load_math_response(input_path, qid)
    logging.info(f"Loaded math response: {math_response.name}")

    # Process the response
    result = asyncio.run(
        process_math_response(
            model_response=math_response,
            model_id=model_id,
            max_retries=max_retries,
            max_parallel=max_parallel,
        )
    )

    # Print the result
    print(f"Original problem: {result.problem}")
    print(f"Original solution: {result.solution}")
    print(f"Model answer: {result.model_answer}")
    print(f"Correctness: {result.correctness_is_correct}")
    print(f"Classification: {result.correctness_classification}")
    print(f"Explanation: {result.correctness_explanation}")

    # Save the result
    save_result(result, output_path_obj, model_id)


if __name__ == "__main__":
    main()
