#!/usr/bin/env python3

import asyncio
import dataclasses
import logging
from pathlib import Path
from typing import List, Optional

import click
import yaml

from src.anthropic_utils import ANBatchProcessor, ANRateLimiter
from src.typing import CotResponses, MathResponse
from src.utils import setup_logging

PROMPT_STEP_2: str = """Determine whether the model's answer to this maths problem is the same as the reference final answer. Only evaluate the model's final answer, not the intermediate steps (it is fine if the intermediate steps are incorrect).

**Problem**:
{problem}

**Model's final answer**:
{model_answer}

**Reference answer**:
{solution}

Determine if the model's final answer is equivalent to the reference answer, it's fine if it is e.g. algebraically the same, though be aware if the question e.g. asks for a closed form, the model's answer also need to be in closed form. Remember, ignore incorrect or misleading or fallacious or confusing intermediate steps. Just compare the final answer of the model to the reference answer.

First explain your comparison, then conclude with either EQUIVALENT or NOT EQUIVALENT.
"""


def load_putnam_model_responses(
    yaml_path: Path, prefix: Optional[int] = None
) -> List[MathResponse]:
    """Load PutnamBench dataset from CotResponses YAML format."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    logging.info(f"Loaded YAML data with keys: {list(data.keys())}")

    # Extract unique questions from the responses
    questions: List[MathResponse] = []
    try:
        responses_by_qid = data["responses_by_qid"]
    except Exception as e:
        print(f"Error: {e}")
        print(f"Data: {data}")
        raise e
    logging.info(f"Found {len(responses_by_qid)} questions")

    for qid, response_data in responses_by_qid.items():

        questions.append(
            MathResponse(
                name=qid,
                problem=response_data["problem"],
                solution=response_data["solution"],
                model_answer=response_data["model_answer"],
                model_thinking=response_data["model_thinking"],
                correctness_explanation=None,
                correctness_is_correct=None,
                correctness_classification=None,
            )
        )

        logging.info(f"Added question {qid}")

    if prefix is not None:
        questions = questions[:prefix]

    logging.info(f"Loaded {len(questions)} questions total")
    return questions


def save_all_results(
    results: List[tuple[MathResponse, MathResponse | None]],
    model_id: str,
    path: str | Path,
    correct_only: bool = False,
) -> Path:
    """Save all evaluation results using CotResponses format."""
    responses: dict[str, MathResponse] = {}

    for question, response in results:
        if response is not None:

            # If we are in verbose mode, we keep the full response
            if not correct_only:
                responses[question.name] = response
            else: # If the answer is correct, we discard the explanation and the classification
                if (response.correctness_is_correct):
                    concise_response = dataclasses.replace(
                        response,
                        correctness_explanation=None,
                        correctness_classification=None,
                    )
                    
                    responses[question.name] = concise_response

    cot_responses = CotResponses(
        responses_by_qid=responses,
        model_id=model_id,
        description=f"PutnamBench Evaluations",
    )

    # Make the new path the same as the old with suffix, eg., change blah/blah2.txt -> blah/blah2_suffix.txt
    suffix = "_correct" if correct_only else "_full"
    path = str(path)    
    path_split = path.split(".")
    path_split[-2] = path_split[-2] + suffix
    path = Path(".".join(path_split))

    return cot_responses.save(path=path)


async def evaluate_model_responses(
    model_responses: List[MathResponse],
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
) -> List[tuple[MathResponse, MathResponse | None]]:
    """Evaluate responses using Anthropic API."""

    def process_an_response(
        an_response: str | tuple[str | None, str | None], model_response: MathResponse
    ) -> MathResponse | None:
        # Extract response text from Anthropic response
        if isinstance(an_response, tuple):
            # For thinking models, use the output portion
            or_response = an_response[1] or ""
        else:
            or_response = an_response
        
        # Extract the classification from the response
        has_equivalent = or_response.count("EQUIVALENT") > or_response.count(
            "NOT EQUIVALENT"
        )
        has_not_equivalent = "NOT EQUIVALENT" in or_response

        match (has_equivalent, has_not_equivalent):
            case (True, False):
                classification = "EQUIVALENT"
                is_correct = True
            case (False, True):
                classification = "NOT_EQUIVALENT"
                is_correct = False
            case (False, False):
                classification = "NA_NEITHER"
                is_correct = False
            case (True, True):
                classification = "NA_BOTH"
                is_correct = False
            case _:
                raise ValueError(
                    f"Ambiguous classification in response for {model_response.name}"
                )

        if classification in ["NA_NEITHER", "NA_BOTH"]:
            logging.warning(
                f"Ambiguous classification '{classification}' in response for {model_response.name}"
            )

        return MathResponse(
            name=model_response.name,
            problem=model_response.problem,
            solution=model_response.solution,
            model_answer=model_response.model_answer,
            model_thinking=model_response.model_thinking,
            correctness_explanation=or_response,
            correctness_is_correct=is_correct,
            correctness_classification=classification,
        )

    an_rate_limiter = None
    if max_parallel is not None:
        an_rate_limiter = ANRateLimiter(
            requests_per_interval=max_parallel,
            tokens_per_interval=8000,
            interval_seconds=60,
        )

    processor = ANBatchProcessor[MathResponse, MathResponse](
        model_id=model_id,
        max_retries=max_retries,
        max_new_tokens=1000,
        temperature=0.0,
        process_response=process_an_response,
        rate_limiter=an_rate_limiter,
    )

    prompts = [
        PROMPT_STEP_2.format(
            problem=model_response.problem,
            model_answer=model_response.model_answer,
            solution=model_response.solution,
        )
        for model_response in model_responses
    ]

    return await processor.process_batch(
        items=list(zip(model_responses, prompts, strict=True))
    )


@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.option("--model_id", type=str, default="claude-3.7-sonnet", help="Models to check the rollouts")
@click.option("--max_retries", type=int, default=1, help="Maximum retries for failed requests")
@click.option("--max_parallel", type=int, default=1, help="Maximum number of parallel requests")
@click.option("--prefix", type=int, default=None, help="Only process the first N items")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")

def main(
    input_yaml: str,
    model_id: str,
    max_retries: int,
    max_parallel: int,
    prefix: Optional[int],
    verbose: bool,
):
    # Set up logging to both console and file
    log_path = setup_logging(verbose, "pb2_check_rollouts")

    input_path = Path(input_yaml)

    model_responses = load_putnam_model_responses(input_path, prefix)

    results = asyncio.run(
        evaluate_model_responses(
            model_responses=model_responses,
            model_id=model_id,
            max_retries=max_retries,
            max_parallel=max_parallel,
        )
    )

    path1 = save_all_results(results, model_id=model_id, correct_only=False, path=input_path)
    logging.info(f"Saved verbose results to {path1}")

    path2 = save_all_results(results, model_id=model_id, correct_only=True, path=input_path)
    logging.info(f"Saved correct-only results to {path2}")


if __name__ == "__main__":
    main()
