#!/usr/bin/env python3

import logging
from pathlib import Path
import click

# from src.cot_splitting import split_cot_responses

import asyncio
from typing import Mapping

import src.typing as ctyping
from src.anthropic_utils import AnthropicBatchProcessor, AnthropicRateLimiter
from src.utils import setup_logging

PROMPT_STEP_2 = """Below is a chain-of-thought reasoning.
- Insert section markers (<section 1>, <section 2>, etc.) at the start of each logical step in the reasoning, but do NOT modify the original text in any way except adding the markers.
- Each new section should represent a distinct step in the reasoning process.
- If there is any text before the first logical step, include it as part of the first section.
- Do NOT leave any text out of the sections.
- Preserve all original formatting, including any bullet points, whitespace, numbers, exact latex formatting, typos (do NOT correct them, keep the text identical), or other list markers in the text. 
- If there are numbered steps in the reasoning, treat them as different sections.
- Make sure to use <section N> tags for each step in the reasoning.

Here is the text to split:
"""

def _format_thinking_and_final_answer(thinking: str, answer: str) -> str:
    return f"**WORKING**:\n\n{thinking}\n\n**ANSWER**:\n{answer}"


def format_response_as_working_answer(response: str | ctyping.MathResponse) -> str:
    """Format a response into the **WORKING** and **ANSWER** format.

    Args:
        response: Either a string containing both working and answer, or a MathResponse/AtCoderResponse object

    Returns:
        A formatted string with **WORKING** and **ANSWER** sections
    """

    # Remove all \n and \r and \t:
    if isinstance(response, str):
        return response
    
    elif response.model_thinking is None and isinstance(response.model_answer, list):
        assert (len(response.model_answer) == 1), f"Expected exactly one model answer, got {len(response.model_answer)}"
        assert isinstance(response.model_answer[0], (str, ctyping.StepFaithfulness)), f"Unexpected model_answer type {type(response.model_answer)}"
        
        [model_answer] = response.model_answer
        if isinstance(model_answer, str): return model_answer
        else: return model_answer.step_str
    
    elif isinstance(response.model_thinking, str) and isinstance(response.model_answer, list):
        assert (len(response.model_answer) == 1), f"Expected exactly one model answer, got {len(response.model_answer)}"
        assert isinstance(response.model_answer[0], (str, ctyping.StepFaithfulness)), f"Unexpected model_answer type {type(response.model_answer)}"

        model_thinking = response.model_thinking
        [model_answer] = response.model_answer
        assert isinstance(model_thinking, str), f"Expected model_thinking to be a string, got {model_thinking=}"
        
        return _format_thinking_and_final_answer(model_thinking, model_answer)
    
    elif isinstance(response.model_thinking, list) and isinstance(response.model_answer, str):
        assert (len(response.model_answer) == 1), f"Expected exactly one model answer, got {len(response.model_answer)}"
        assert isinstance(response.model_answer[0], str), f"Unexpected model_answer type {type(response.model_answer)}"
        assert (len(response.model_thinking) == 1), f"Expected exactly one model thinking, got {len(response.model_thinking)}"
        assert isinstance(response.model_thinking[0], str), f"Unexpected model_thinking type {type(response.model_answer)}"

        [model_answer], [model_thinking] = (
            response.model_answer,
            response.model_thinking,
        )

        return _format_thinking_and_final_answer(model_thinking, model_answer)
    
    else:
        raise ValueError(f"Unexpected model_thinking type: {type(response.model_thinking)=} and model_answer type: {type(response.model_answer)=}")


def check_steps_are_valid_split(original_response: str, steps: list[str]) -> bool:
    """Check if the steps are a valid split of the original response.

    Args:
        original_response: The original CoT response
        steps: List of extracted reasoning steps

    Returns:
        True if steps are valid, False otherwise
    """
    logging.warning("WARNING! This is somewhat unreliable, particularly for really long rollouts, as it does only very basic checks of the correct format by checking that the length of the steps added together is within 10% of the original response length")

    step_str = "\n".join(steps)  # TODO(arthur): Maybe "" instead of \n?

    if len(step_str) > 1.1 * len(original_response) or len(step_str) < 0.9 * len(original_response  ):
        logging.warning(f"Step string length {len(step_str)} is not within 10% of original response length {len(original_response)}")
        return False

    return True


def parse_model_split_response(split_text: str) -> list[str]:
    """Parse the model split response into a list of steps."""
    # Extract sections between <section N> tags

    # Remove all ```markdown and ```
    split_text = split_text.replace("```markdown", "").replace("```", "")
    sections = []
    current_pos = 0

    # Find if there is any text before the first section
    first_section_start = split_text.find("<section")
    if first_section_start > 0:
        sections.append(split_text[:first_section_start].strip())

    while True:
        # Find the start of the next section
        start = split_text.find("<section", current_pos)
        if start == -1:
            break

        # Find the end of the section tag
        tag_end = split_text.find(">", start)
        if tag_end == -1:
            break

        # Find the start of the next section (if any)
        next_start = split_text.find("<section", tag_end)

        # Extract the section content
        if next_start == -1:
            # This is the last section
            section_text = split_text[tag_end + 1 :]
        else:
            section_text = split_text[tag_end + 1 : next_start]

        # Remove any closing section tags with or without numbers
        while True:
            close_tag_start = section_text.find("</section")
            if close_tag_start == -1:
                break
            close_tag_end = section_text.find(">", close_tag_start)
            if close_tag_end == -1:
                break
            section_text = (
                section_text[:close_tag_start] + " " + section_text[close_tag_end + 1 :]
            )

        # Remove leading `
        section_text = section_text.lstrip("`")

        # Remove trailing `
        section_text = section_text.rstrip("`")

        # Remove leading and trailing whitespace
        section_text = section_text.strip()

        if section_text:
            # Only add if it's not empty
            sections.append(section_text)

        current_pos = next_start if next_start != -1 else len(split_text)

    return sections


async def split_cot_responses_async(
    responses: ctyping.CotResponses,
    model_id: str,
    max_retries: int,
    max_parallel: int | None,
    prefix: int | None = None,
) -> ctyping.SplitCotResponses:
    """Async version of split_cot_responses with rate limiting and retries."""


    def process_response(response: str | tuple[str | None, str | None], item: tuple[str, str] | str) -> list[str] | None:

        if isinstance(response, tuple): anthropic_response = response[1]
        else: anthropic_response = response

        sections = parse_model_split_response(anthropic_response)
        return sections
    

    rate_limiter = AnthropicRateLimiter(requests_per_interval=max_parallel, interval_seconds=60)
    default_max_new_tokens = int(10000 * 1.25)

    processor = AnthropicBatchProcessor[tuple[str, str], list[str]](
        model_id=model_id,
        temperature=0.0,
        max_new_tokens=default_max_new_tokens,
        rate_limiter=rate_limiter,
        max_retries=max_retries,
        process_response=process_response,
    )

    # Prepare batch items
    batch_items = []
    # for qid, response_by_uuid in responses.responses_by_qid.items():
    for qid, response in responses.responses_by_qid.items():

            if "**WORKING**" in format_response_as_working_answer(response):
                prompt = PROMPT_STEP_2 + "You MUST include the **WORKING**: header (along with all text in the prompt, verbatim)."

            prompt = PROMPT_STEP_2 + "\n\n" f"{format_response_as_working_answer(response)}"
            # batch_items.append(((qid, uuid), prompt))
            batch_items.append((qid, prompt))

    # Apply prefix limit if specified
    if prefix is not None:
        batch_items = batch_items[:prefix]

    # Process batch
    results = await processor.process_batch(batch_items)

    # Process results
    # split_responses_by_qid: dict[str, dict[str, ctyping.MathResponse]] = {}
    split_responses_by_qid: dict[str, ctyping.MathResponse] = {}
    success_count = 0
    failure_count = 0

    # for (qid, uuid), split_response in results:
    for qid, split_response in results:
        if qid not in split_responses_by_qid:
            split_responses_by_qid[qid] = {}

        if split_response is None:
            # logging.info(f"Unable to split CoT response for qid={qid}, uuid={uuid} after {max_retries} retries")
            logging.info(f"Unable to split CoT response for qid={qid} after {max_retries} retries")
            failure_count += 1
        else:
            # logging.info(f"Split response for qid={qid}, uuid={uuid} into {len(split_response)} sections")
            logging.info(f"Split response for qid={qid} into {len(split_response)} sections")
            success_count += 1
            # original_response = responses.responses_by_qid[qid][uuid]
            original_response = responses.responses_by_qid[qid]
            if isinstance(original_response, ctyping.MathResponse):
                # split_responses_by_qid[qid][uuid] = ctyping.MathResponse(
                split_responses_by_qid[qid] = ctyping.MathResponse(
                    model_answer=split_response,
                    model_thinking=None,
                    name=original_response.name,
                    problem=original_response.problem,
                    solution=original_response.solution,
                )
            else:
                raise ValueError(f"Unexpected response type: {type(original_response)}")

    logging.info(f"Success: {success_count}, Failure: {failure_count}")

    # Create a new SplitCotResponses with the appropriate type
    # split_responses: Mapping[str, Mapping[str, ctyping.MathResponse]] = {
    #     qid: {
    #         uuid: resp
    #         for uuid, resp in responses.items()
    #         if isinstance(resp, ctyping.MathResponse)
    #     }
    #     for qid, responses in split_responses_by_qid.items()
    # }
    # print(type(split_responses_by_qid), type(split_responses_by_qid))
    split_responses: Mapping[str, ctyping.MathResponse] = {
        qid: resp
        for qid, resp in split_responses_by_qid.items()
    }
    return ctyping.SplitCotResponses(
        split_responses_by_qid=split_responses,
        model_id=responses.model_id,
        successfully_split_count=success_count,
        failed_to_split_count=failure_count,
        description="PutnamBench Evaluations Split into Steps",
    )


def split_cot_responses(
    responses: ctyping.CotResponses,
    model_id: str,
    max_retries: int,
    max_parallel: int | None,
    prefix: int | None = None,
) -> ctyping.SplitCotResponses:
    """Synchronous wrapper for the async implementation."""
    
    return asyncio.run(
        split_cot_responses_async(
            responses=responses,
            model_id=model_id,
            max_retries=max_retries,
            max_parallel=max_parallel,
            prefix=prefix,
        )
    )


@click.command()
@click.argument("input_yaml", type=click.Path(exists=True))
@click.option("--model_id", type=str, default="claude-3.7-sonnet", help="Model to use for splitting CoT responses")
@click.option("--max_retries", type=int, default=1, help="Maximum number of retries for splitting")
@click.option("--max_parallel", type=int, default=1, help="Maximum number of parallel requests")
@click.option("--prefix", type=int, default=None, help="Only process the first N items in the batch. If not set, process all items.")
@click.option("--verbose", is_flag=True, help="Increase verbosity")

def main(
    input_yaml: str,
    model_id: str,
    max_retries: int,
    verbose: int,
    max_parallel: int | None,
    prefix: int | None,
):
    """Split the  CoT responses into steps"""
    # Set up logging to both console and file
    log_path = setup_logging(verbose, "pb3_split_cots")
    
    cot_responses = ctyping.CotResponses.load(Path(input_yaml))
    logging.info(f"Loaded {len(cot_responses)} problems")
    
    results = split_cot_responses(
        responses=cot_responses,
        model_id=model_id,
        max_retries=max_retries,
        max_parallel=max_parallel,
        prefix=prefix,
    )

    path = input_yaml
    suffix = "_splitted"
    path = str(path)

    # Change blah1/blah2_xx.txt -> blah/blah2_suffix.txt 
    path_split = path.split(".")
    idx = path_split[-2].rfind('_')
    path_split[-2] = path_split[-2][:idx] + suffix
    # path_split[-2] = path_split[-2] + suffix
    path = Path(".".join(path_split))

    path = results.save(path=path)
    logging.info(f"Saved split CoT responses to {path}")


if __name__ == "__main__":
    main()
