#!/usr/bin/env python3

"""Process a single MathResponse object using OpenRouter API.

Example usage:
python -m dotenv run python scripts/putnam/process_single_math_response.py \
    --model_id "anthropic/claude-3.5-sonnet" \
    --verbose
"""

import asyncio
import logging
from typing import Optional

import click
from beartype import beartype

from chainscope.api_utils.open_router_utils import ORBatchProcessor, ORRateLimiter
from chainscope.typing import MathResponse

_EVALUATION_PROMPT: str = """Determine whether the model's answer to this maths problem is the same as the reference final answer. Only evaluate the model's final answer, not the intermediate steps (it is fine if the intermediate steps are incorrect).

**Problem**:
{problem}

**Model's final answer**:
{model_answer}

**Reference answer**:
{solution}

Determine if the model's final answer is equivalent to the reference answer, it's fine if it is e.g. algebraically the same, though be aware if the question e.g. asks for a closed form, the model's answer also need to be in closed form. Remember, ignore incorrect or misleading or fallacious or confusing intermediate steps. Just compare the final answer of the model to the reference answer.

First explain your comparison, then conclude with either EQUIVALENT or NOT EQUIVALENT.
"""


@beartype
def process_or_response(
    or_response: str | tuple[str | None, str | None], model_response: MathResponse
) -> MathResponse:
    """Process the response from OpenRouter API.

    Args:
        or_response: The response from OpenRouter API
        model_response: The original MathResponse object

    Returns:
        Updated MathResponse object with correctness information
    """
    # Extract the text content from the response
    if isinstance(or_response, tuple):
        # If we got a tuple (thinking, answer), use just the answer part
        if or_response[1] is not None:
            response_text = or_response[1]
        elif or_response[0] is not None:
            response_text = or_response[0]
        else:
            response_text = ""
    else:
        response_text = or_response

    # Extract the classification from the response
    has_equivalent = response_text.count("EQUIVALENT") > response_text.count(
        "NOT EQUIVALENT"
    )
    has_not_equivalent = "NOT EQUIVALENT" in response_text

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

    # Create updated response
    return MathResponse(
        name=model_response.name,
        problem=model_response.problem,
        solution=model_response.solution,
        model_answer=model_response.model_answer,
        model_thinking=model_response.model_thinking,
        correctness_explanation=response_text,
        correctness_is_correct=is_correct,
        correctness_classification=classification,
    )


@beartype
async def process_math_response(
    model_response: MathResponse,
    model_id: str,
    max_retries: int = 1,
    max_parallel: Optional[int] = None,
    temperature: float = 0.0,
) -> MathResponse:
    """Process a MathResponse object using OpenRouter API.

    Args:
        model_response: The MathResponse object to process
        model_id: The model ID to use for evaluation
        max_retries: Maximum number of retries for failed requests
        max_parallel: Maximum number of parallel requests
        temperature: Temperature for text generation

    Returns:
        Updated MathResponse object with correctness information
    """
    # Create rate limiter if max_parallel is specified
    or_rate_limiter = None
    if max_parallel is not None:
        or_rate_limiter = ORRateLimiter(
            requests_per_interval=max_parallel,
            interval_seconds=1,
        )

    # Create batch processor
    processor = ORBatchProcessor[MathResponse, MathResponse](
        model_id=model_id,
        max_retries=max_retries,
        max_new_tokens=1000,
        temperature=temperature,
        process_response=process_or_response,
        rate_limiter=or_rate_limiter,
    )

    # Get model answer as string
    if isinstance(model_response.model_answer, list):
        if len(model_response.model_answer) > 0:
            if isinstance(model_response.model_answer[0], str):
                model_answer = model_response.model_answer[0]
            else:
                # This is probably a StepFaithfulness object
                model_answer = str(model_response.model_answer[0])
        else:
            model_answer = ""
    else:
        model_answer = str(model_response.model_answer)

    # Format prompt
    prompt = _EVALUATION_PROMPT.format(
        problem=model_response.problem,
        model_answer=model_answer,
        solution=model_response.solution,
    )

    # Process the response
    results = await processor.process_batch(items=[(model_response, prompt)])

    # Return the processed response
    if results and results[0][1] is not None:
        return results[0][1]
    else:
        # Return original response with error indication
        return MathResponse(
            name=model_response.name,
            problem=model_response.problem,
            solution=model_response.solution,
            model_answer=model_response.model_answer,
            model_thinking=model_response.model_thinking,
            correctness_explanation="Error processing response",
            correctness_is_correct=False,
            correctness_classification="NA_NEITHER",
        )


@click.command()
@click.option(
    "--model_id",
    "-s",
    type=str,
    default="anthropic/claude-3.5-sonnet",
    help="Models for evaluation",
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
    model_id: str,
    max_retries: int,
    max_parallel: Optional[int],
    verbose: bool,
):
    """Example of using process_math_response with a sample MathResponse."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Sample MathResponse for testing
    sample_response = MathResponse(
        name="test_problem",
        problem="Calculate 2 + 2",
        solution="4",
        model_answer=["The answer is 4"],
        model_thinking="To calculate 2 + 2, I add the numbers 2 and 2 together, which equals 4.",
        correctness_explanation=None,
        correctness_is_correct=None,
        correctness_classification=None,
    )

    # Process the response
    result = asyncio.run(
        process_math_response(
            model_response=sample_response,
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


if __name__ == "__main__":
    main()
