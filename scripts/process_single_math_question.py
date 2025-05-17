#!/usr/bin/env python3

import logging
import os

from anthropic import AsyncAnthropic
from beartype import beartype

from chainscope.api_utils.anthropic_utils import (
    ANTHROPIC_MODEL_ALIASES,
    MAX_THINKING_TIMEOUT,
    get_budget_tokens,
    is_anthropic_thinking_model,
)
from chainscope.typing import MathQuestion, MathResponse


@beartype
async def process_math_question_anthropic(
    question: MathQuestion,
    model_id: str = "claude-3-sonnet",
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_new_tokens: int = 4096,
    max_retries: int = 3,
    preamble: str = "Solve this math problem step-by-step, reasoning first and then producing an answer.\n\n",
) -> MathResponse:
    """
    Process a single MathQuestion and return the model's response using Anthropic models.

    Args:
        question: MathQuestion object containing the problem
        model_id: Anthropic model ID (default: claude-3-sonnet)
        temperature: Temperature for text generation
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum number of tokens to generate
        max_retries: Maximum number of retry attempts
        preamble: Text to add before the problem statement

    Returns:
        MathResponse object containing the model's thinking and answer
    """
    # Check if ANTHROPIC_API_KEY is set
    assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"

    # Create Anthropic client
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Check if this is a thinking model
    is_thinking_model = is_anthropic_thinking_model(model_id)
    thinking_budget_tokens = get_budget_tokens(model_id) if is_thinking_model else None

    # Get the actual model name from aliases if needed
    base_model_id = model_id.split("/")[-1].split("_")[0]
    actual_model_id = ANTHROPIC_MODEL_ALIASES.get(base_model_id, base_model_id)

    # Create the full prompt
    prompt = f"{preamble}{question.problem}"
    logging.info(f"Running prompt:\n{prompt}")

    # Variables to store the thinking and answer
    thinking = None
    answer = None

    # Try multiple times if specified
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(f"Retry attempt {attempt} of {max_retries}")

            # Set up message creation parameters
            create_params = {
                "model": actual_model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

            # Adjust parameters for thinking models
            if is_thinking_model:
                assert thinking_budget_tokens is not None
                create_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }
                # Temperature can only be set to 1 for thinking models
                create_params["temperature"] = 1
                # Top-p must be unset for thinking models
                del create_params["top_p"]
                # `max_tokens` must be greater than `thinking.budget_tokens`
                create_params["max_tokens"] = max_new_tokens + thinking_budget_tokens
                # Set timeout for thinking models
                create_params["timeout"] = MAX_THINKING_TIMEOUT

            # Make the API call
            response = await client.messages.create(**create_params)

            # Check if we got a valid response
            if not response or not response.content or len(response.content) == 0:
                logging.warning("Empty response content")
                continue

            # Extract content based on content types
            if len(response.content) == 1 and response.content[0].type == "text":
                # For regular model responses
                full_response = response.content[0].text
                thinking = None
                answer = full_response.strip()
            elif (
                len(response.content) == 2
                and response.content[0].type == "thinking"
                and response.content[1].type == "text"
            ):
                # For thinking model responses
                thinking = response.content[0].thinking
                answer = response.content[1].text.strip()
                logging.info(
                    f"Token usage breakdown for {model_id}:\n"
                    f"  Total tokens: {response.usage.output_tokens}\n"
                )
            else:
                logging.warning(f"Unexpected response structure: {response.content}")
                continue

            # If we have an answer, break out of the retry loop
            if answer is not None:
                logging.info("Found valid result!")
                break

        except Exception as e:
            if attempt == max_retries - 1:
                logging.warning(f"Failed after {max_retries} retries: {str(e)}")
                raise
            logging.warning(f"Error on attempt {attempt + 1}: {str(e)}, retrying...")

    # Create the MathResponse
    math_response = MathResponse(
        name=question.name,
        problem=question.problem,
        solution=question.solution,
        model_thinking=thinking,
        model_answer=[answer] if answer else ["Failed to generate an answer"],
    )

    # Close the client
    await client.close()

    return math_response


# Example usage:
# async def main():
#     question = MathQuestion(
#         name="putnam_2000_a1",
#         problem="Prove that...",
#         solution="Solution is..."
#     )
#     response = await process_math_question_anthropic(
#         question,
#         model_id="claude-3-sonnet"  # or "claude-3.7-sonnet_32k" for thinking model
#     )
#     print(f"Thinking: {response.model_thinking}")
#     print(f"Answer: {response.model_answer[0]}")
#
# if __name__ == "__main__":
#     asyncio.run(main())
