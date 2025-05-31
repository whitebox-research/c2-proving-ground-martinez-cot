import asyncio
import base64
import logging
import os
import time
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Generic, TypeVar

import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types.message_create_params import \
    MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.text_block import TextBlock
from anthropic.types.thinking_block import ThinkingBlock
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from chainscope.api_utils.batch_processor import (BatchItem, BatchProcessor,
                                                  BatchResult)
from chainscope.typing import (AnthropicBatchInfo, DatasetParams,
                               QuestionResponseId, SamplingParams)
from scripts.view_faithfulness_dataset import Response

load_dotenv()

# Type variables for generic batch processor
T = TypeVar('T')
U = TypeVar('U')

ANTHROPIC_MODEL_ALIASES = {
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3.6-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
}


MAX_THINKING_TIMEOUT = 7 * 60  # 5 minutes
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
assert ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set"
@dataclass
class AnthropicLimits:
    requests_limit: int
    requests_remaining: int
    requests_reset: str
    tokens_limit: int
    tokens_remaining: int
    tokens_reset: str
    input_tokens_limit: int
    input_tokens_remaining: int
    input_tokens_reset: str
    output_tokens_limit: int
    output_tokens_remaining: int
    output_tokens_reset: str
    retry_after: float | None = None
    org_tpm_remaining: int = 80000  # Organization tokens per minute limit
    org_tpm_reset: str = ""  # When the org TPM limit resets


def get_anthropic_limits() -> AnthropicLimits:
    """Extract rate limits from Anthropic API response headers."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.with_raw_response.create(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
    )
    return AnthropicLimits(
        requests_limit=int(
            response.headers.get("anthropic-ratelimit-requests-limit", 0)
        ),
        requests_remaining=int(
            response.headers.get("anthropic-ratelimit-requests-remaining", 0)
        ),
        requests_reset=response.headers.get("anthropic-ratelimit-requests-reset", ""),
        tokens_limit=int(response.headers.get("anthropic-ratelimit-tokens-limit", 0)),
        tokens_remaining=int(
            response.headers.get("anthropic-ratelimit-tokens-remaining", 0)
        ),
        tokens_reset=response.headers.get("anthropic-ratelimit-tokens-reset", ""),
        input_tokens_limit=int(
            response.headers.get("anthropic-ratelimit-input-tokens-limit", 0)
        ),
        input_tokens_remaining=int(
            response.headers.get("anthropic-ratelimit-input-tokens-remaining", 0)
        ),
        input_tokens_reset=response.headers.get(
            "anthropic-ratelimit-input-tokens-reset", ""
        ),
        output_tokens_limit=int(
            response.headers.get("anthropic-ratelimit-output-tokens-limit", 0)
        ),
        output_tokens_remaining=int(
            response.headers.get("anthropic-ratelimit-output-tokens-remaining", 0)
        ),
        output_tokens_reset=response.headers.get(
            "anthropic-ratelimit-output-tokens-reset", ""
        ),
        retry_after=float(response.headers.get("retry-after", 0))
        if "retry-after" in response.headers
        else None,
        org_tpm_remaining=int(
            response.headers.get("anthropic-ratelimit-org-tpm-remaining", 80000)
        ),
        org_tpm_reset=response.headers.get("anthropic-ratelimit-org-tpm-reset", ""),
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_anthropic_requests_{timestamp}.log"
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

@dataclass
class ANRateLimiter:
    requests_per_interval: int
    tokens_per_interval: int
    interval_seconds: int
    input_tokens: float = field(init=False)
    output_tokens: float = field(init=False)
    requests: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False)
    client: Anthropic = field(default_factory=Anthropic)
    org_tpm_limit: int = 80000
    org_tpm_usage: float = field(init=False)
    org_tpm_last_update: float = field(init=False)

    def __post_init__(self):
        self.input_tokens = self.tokens_per_interval
        self.output_tokens = self.tokens_per_interval
        self.requests = self.requests_per_interval
        self.last_update = time.time()
        self._lock = asyncio.Lock()
        self.org_tpm_usage = 0
        self.org_tpm_last_update = time.time()
        

        logging.info(
            f"ANRateLimiter initialized with {self.requests_per_interval} requests, "
            f"{self.tokens_per_interval} tokens per {self.interval_seconds} seconds, "
            f"and org TPM limit of {self.org_tpm_limit}"
        )

    async def acquire(self, prompt: str, model: str):
        async with self._lock:
            # Everything else was too complicated to implement
            await asyncio.sleep(4)

    def update_token_usage(self, output_tokens: int):
        """Update output token count after receiving a response"""
        self.output_tokens = max(0, self.output_tokens - output_tokens)
        # Update org TPM usage with actual tokens used
        now = time.time()
        time_passed = now - self.org_tpm_last_update
        self.org_tpm_usage = max(
            0,
            self.org_tpm_usage * (1 - time_passed / 60),  # Decay over 1 minute
        )
        self.org_tpm_usage += output_tokens
        self.org_tpm_last_update = now

    async def acquire_with_backoff(self, prompt: str, model: str, max_retries: int = 3):
        """Acquire rate limit with exponential backoff retry logic"""
        ## UPDATE: made it multiplicative instead of exponential
        import random  # Add at top of file if not already present

        for attempt in range(max_retries):
            try:
                await self.acquire(prompt, model)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (60*attempt) + random.uniform(0, 1)
                logging.warning(
                    f"Rate limit acquisition failed, retrying in {wait_time:.2f}s: {str(e)}"
                )
                await asyncio.sleep(wait_time)


def is_anthropic_thinking_model(model_id: str) -> bool:
    return "claude-3.7-sonnet" in model_id and "_" in model_id


def get_budget_tokens(model_id: str) -> int:
    """Get the budget tokens for a given model ID."""
    if "_" not in model_id:
        raise ValueError(
            f"Model id {model_id} does not have underscore for indicating budget tokens"
        )

    budget_part = model_id.split("_")[-1]
    if budget_part == "1k":
        return 1024
    elif budget_part == "10k":
        return 10000
    elif budget_part == "32k":
        return 32000
    elif budget_part == "64k":
        return 64000
    else:
        try:
            if "k" in budget_part:
                return int(budget_part.replace("k", "")) * 1000
            else:
                return int(budget_part)
        except ValueError:
            raise ValueError(
                f"Invalid budget tokens for model {model_id}. "
                f"Expected an integer, with optional `k` suffix, got `{budget_part}`"
            )

async def generate_an_response_async(
    prompt: str,
    model_id: str,
    client: AsyncAnthropic,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    get_result_from_response: Callable[
        [str | tuple[str | None, str | None]], Any | None
    ],
    rate_limiter: ANRateLimiter | None = None,
) -> Any | None:
    """Generate a response from an Anthropic model.

    Args:
        prompt: The prompt to run on the model
        model_id: The model ID to call
        client: The Anthropic client
        temperature: Temperature parameter for generation
        max_new_tokens: Maximum number of new tokens to generate
        max_retries: Maximum number of retry attempts for each model
        get_result_from_response: Callback that processes the model response and returns
            either a valid result or None if the response should be retried

    Returns:
        Processed result or None if all attempts failed
    """
    logging.info(f"generate_an_response_async called with initial model_id: {model_id}")
    logging.info(f"Prompt length: {len(prompt)}")

    is_thinking_model = is_anthropic_thinking_model(model_id)
    thinking_budget_tokens = get_budget_tokens(model_id) if is_thinking_model else None
    logging.info(f"Is thinking model: {is_thinking_model}, Budget tokens: {thinking_budget_tokens}")

    model_id = model_id.split("/")[-1].split("_")[0]
    model_id = ANTHROPIC_MODEL_ALIASES[model_id]

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(
                    f"Retry attempt {attempt} of {max_retries} for generating a response"
                )

            if rate_limiter:
                await rate_limiter.acquire_with_backoff(prompt, model_id)

            create_params = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_new_tokens,
            }

            if is_thinking_model:
                assert thinking_budget_tokens is not None
                create_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }
                # Temperature can only be set to 1 for thinking models
                create_params["temperature"] = 1
                # `max_tokens` must be greater than `thinking.budget_tokens`
                create_params["max_tokens"] = max_new_tokens + thinking_budget_tokens
                # Set the timeout explicitly so that Anthropic doesn't complain about
                # this request _potentially_ taking too long
                create_params["timeout"] = MAX_THINKING_TIMEOUT

            # Use acreate instead of create for async operation
            
            # start_time = time.perf_counter()
            an_response = await client.messages.create(**create_params)
            # end_time = time.perf_counter()

            if rate_limiter:
                logging.info(f"Got usage: {an_response.usage}")
                rate_limiter.update_token_usage(an_response.usage.output_tokens)

            if (
                not an_response
                or not an_response.content
                or len(an_response.content) == 0
            ):
                logging.info("Invalid response content")
                continue

            if len(an_response.content) == 1 and isinstance(
                an_response.content[0], TextBlock
            ):
                result = get_result_from_response(an_response.content[0].text)
            elif (
                len(an_response.content) == 2
                and isinstance(an_response.content[0], ThinkingBlock)
                and isinstance(an_response.content[1], TextBlock)
            ):
                logging.info(f"Received response thinking model: {an_response.content[1].text}")
                # Log token usage for thinking vs output
                # thinking_tokens = await client.messages.count_tokens(model=model_id, messages=[{"role": "user", "content": an_response.content[0].thinking}])
                # output_tokens = await client.messages.count_tokens(model=model_id, messages=[{"role": "user", "content": an_response.content[1].text}])
                # total_tokens = an_response.usage.output_tokens
                # logging.info(
                #     f"Token usage breakdown for {model_id} for problem {problem_name}:\n"
                #     f"  Thinking tokens: {thinking_tokens.input_tokens}\n"
                #     f"  Output tokens: {output_tokens.input_tokens}\n"
                #     f"  Total tokens: {total_tokens}\n"
                #     f"  Thinking percentage: {(thinking_tokens.input_tokens / total_tokens * 100):.1f}%"
                #     f"  Time taken: {end_time - start_time:.2f} seconds"
                # )
                result = get_result_from_response(
                    (
                        an_response.content[0].thinking,
                        an_response.content[1].text,
                    )
                )

            if result is not None:
                logging.info("Found valid result!")
                return result

            logging.info(
                f"Invalid result on attempt {attempt + 1} for model {model_id}, retrying..."
            )
            continue

        except Exception as e:
            if attempt == max_retries:
                logging.warning(
                    f"Failed to process response after {max_retries} retries "
                    f"for model {model_id}: {str(e)}"
                )
                return None
            logging.warning(
                f"Error on attempt {attempt + 1} for model {model_id}: {str(e)}, retrying..."
            )
            continue

    return None

async def generate_an_response_async_with_text(
    prompt: str,
    problem_name: str,
    model_id: str,
    client: AsyncAnthropic,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    get_result_from_response: Callable[
        [str | tuple[str | None, str | None]], Any | None
    ],
    rate_limiter: ANRateLimiter | None = None,
) -> Any | None:
    """Generate a response from an Anthropic model.

    Args:
        prompt: The prompt to run on the model
        model_id: The model ID to call
        client: The Anthropic client
        temperature: Temperature parameter for generation
        max_new_tokens: Maximum number of new tokens to generate
        max_retries: Maximum number of retry attempts for each model
        get_result_from_response: Callback that processes the model response and returns
            either a valid result or None if the response should be retried

    Returns:
        Processed result or None if all attempts failed
    """
    logging.info(f"generate_an_response_async called with initial model_id: {model_id}")
    logging.info(f"Prompt length: {len(prompt)}")

    is_thinking_model = is_anthropic_thinking_model(model_id)
    thinking_budget_tokens = get_budget_tokens(model_id) if is_thinking_model else None
    logging.info(f"Is thinking model: {is_thinking_model}, Budget tokens: {thinking_budget_tokens}")

    model_id = model_id.split("/")[-1].split("_")[0]
    model_id = ANTHROPIC_MODEL_ALIASES[model_id]

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(
                    f"Retry attempt {attempt} of {max_retries} for generating a response"
                )

            if rate_limiter:
                await rate_limiter.acquire_with_backoff(prompt, model_id)

            create_params = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_new_tokens,
            }

            if is_thinking_model:
                assert thinking_budget_tokens is not None
                create_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }
                # Temperature can only be set to 1 for thinking models
                create_params["temperature"] = 1
                # `max_tokens` must be greater than `thinking.budget_tokens`
                create_params["max_tokens"] = max_new_tokens + thinking_budget_tokens
                # Set the timeout explicitly so that Anthropic doesn't complain about
                # this request _potentially_ taking too long
                create_params["timeout"] = MAX_THINKING_TIMEOUT

            # Use acreate instead of create for async operation
            
            start_time = time.perf_counter()
            an_response = await client.messages.create(**create_params)
            end_time = time.perf_counter()

            if rate_limiter:
                logging.info(f"Got usage: {an_response.usage}")
                rate_limiter.update_token_usage(an_response.usage.output_tokens)

            if (
                not an_response
                or not an_response.content
                or len(an_response.content) == 0
            ):
                logging.info("Invalid response content")
                continue

            if len(an_response.content) == 1 and isinstance(
                an_response.content[0], TextBlock
            ):
                result = get_result_from_response(an_response.content[0].text)
            elif (
                len(an_response.content) == 2
                and isinstance(an_response.content[0], ThinkingBlock)
                and isinstance(an_response.content[1], TextBlock)
            ):
                logging.info(f"Received response thinking model: {an_response.content[1].text}")
                # Log token usage for thinking vs output
                thinking_tokens = await client.messages.count_tokens(model=model_id, messages=[{"role": "user", "content": an_response.content[0].thinking}])
                output_tokens = await client.messages.count_tokens(model=model_id, messages=[{"role": "user", "content": an_response.content[1].text}])
                total_tokens = an_response.usage.output_tokens
                logging.info(
                    f"Token usage breakdown for {model_id} for problem {problem_name}:\n"
                    f"  Thinking tokens: {thinking_tokens.input_tokens}\n"
                    f"  Output tokens: {output_tokens.input_tokens}\n"
                    f"  Total tokens: {total_tokens}\n"
                    f"  Thinking percentage: {(thinking_tokens.input_tokens / total_tokens * 100):.1f}%"
                    f"  Time taken: {end_time - start_time:.2f} seconds"
                )
                result = get_result_from_response(
                    (
                        an_response.content[0].thinking,
                        an_response.content[1].text,
                    )
                )

            if result is not None:
                logging.info("Found valid result!")
                return result

            logging.info(
                f"Invalid result on attempt {attempt + 1} for model {model_id}, retrying..."
            )
            continue

        except Exception as e:
            if attempt == max_retries:
                logging.warning(
                    f"Failed to process response after {max_retries} retries "
                    f"for model {model_id}: {str(e)}"
                )
                return None
            logging.warning(
                f"Error on attempt {attempt + 1} for model {model_id}: {str(e)}, retrying..."
            )
            continue

    return None

def convert_image_to_base64(image_path):        
    # image_path = question.image_path
    # Convert image to base64 if it exists

    image_base64 = None
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64
        
# print(image_base64)



async def generate_an_response_async_with_image(
    prompt: str,
    problem_name: str,
    image_path: str,
    model_id: str,
    client: AsyncAnthropic,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    get_result_from_response: Callable[
        [str | tuple[str | None, str | None]], Any | None
    ],
    rate_limiter: ANRateLimiter | None = None,
) -> Any | None:
    """Generate a response from an Anthropic model.

    Args:
        prompt: The prompt to run on the model
        model_id: The model ID to call
        client: The Anthropic client
        temperature: Temperature parameter for generation
        max_new_tokens: Maximum number of new tokens to generate
        max_retries: Maximum number of retry attempts for each model
        get_result_from_response: Callback that processes the model response and returns
            either a valid result or None if the response should be retried

    Returns:
        Processed result or None if all attempts failed
    """
    logging.info(f"Running prompt:\n{prompt}")

    is_thinking_model = is_anthropic_thinking_model(model_id)
    thinking_budget_tokens = get_budget_tokens(model_id) if is_thinking_model else None

    model_id = model_id.split("/")[-1].split("_")[0]
    model_id = ANTHROPIC_MODEL_ALIASES[model_id]
    
    logging.info(f"Model ID: {model_id}")
    logging.info(f"is_thinking_model: {is_thinking_model}")

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(
                    f"Retry attempt {attempt} of {max_retries} for generating a response"
                )

            if rate_limiter:
                await rate_limiter.acquire_with_backoff(prompt, model_id)

            create_params = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": convert_image_to_base64(image_path),
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature
            }

            if is_thinking_model:
                assert thinking_budget_tokens is not None
                create_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }
                # Temperature can only be set to 1 for thinking models
                create_params["temperature"] = 1
                # `max_tokens` must be greater than `thinking.budget_tokens`
                create_params["max_tokens"] = max_new_tokens + thinking_budget_tokens
                # Set the timeout explicitly so that Anthropic doesn't complain about
                # this request _potentially_ taking too long
                create_params["timeout"] = MAX_THINKING_TIMEOUT

            # Use acreate instead of create for async operation
            start_time = time.perf_counter()
            an_response = await client.messages.create(**create_params)
            end_time = time.perf_counter()
            if rate_limiter:
                logging.info(f"Got usage: {an_response.usage}")
                rate_limiter.update_token_usage(an_response.usage.output_tokens)

            if (
                not an_response
                or not an_response.content
                or len(an_response.content) == 0
            ):
                logging.info("Invalid response content")
                continue

            if len(an_response.content) == 1 and isinstance(
                an_response.content[0], TextBlock
            ):
                logging.info(f"Received response 1: {an_response.content[0].text}")
                result = get_result_from_response(an_response.content[0].text)
            elif (
                len(an_response.content) == 2
                and isinstance(an_response.content[0], ThinkingBlock)
                and isinstance(an_response.content[1], TextBlock)
            ):
                logging.info(f"Received response thinking model: {an_response.content[1].text}")
                # Log token usage for thinking vs output
                thinking_tokens = await client.messages.count_tokens(model=model_id, messages=[{"role": "user", "content": an_response.content[0].thinking}])
                output_tokens = await client.messages.count_tokens(model=model_id, messages=[{"role": "user", "content": an_response.content[1].text}])
                total_tokens = an_response.usage.output_tokens
                logging.info(
                    f"Token usage breakdown for {model_id} for problem {problem_name}:\n"
                    f"  Thinking tokens: {thinking_tokens.input_tokens}\n"
                    f"  Output tokens: {output_tokens.input_tokens}\n"
                    f"  Total tokens: {total_tokens}\n"
                    f"  Thinking percentage: {(thinking_tokens.input_tokens / total_tokens * 100):.1f}%"
                    f"  Time taken: {end_time - start_time:.2f} seconds"
                )
                result = get_result_from_response(
                    (
                        an_response.content[0].thinking,
                        an_response.content[1].text,
                    )
                )

            if result is not None:
                logging.info("Found valid result!")
                return result

            logging.info(
                f"Invalid result on attempt {attempt + 1} for model {model_id}, retrying..."
            )
            continue

        except Exception as e:
            if attempt == max_retries:
                logging.warning(
                    f"Failed to process response after {max_retries} retries "
                    f"for model {model_id}: {str(e)}"
                )
                return None
            logging.warning(
                f"Error on attempt {attempt + 1} for model {model_id}: {str(e)}, retrying..."
            )
            continue

    return None

class ANBatchProcessor(BatchProcessor[BatchItem, BatchResult]):
    def __init__(
        self,
        model_id: str,
        temperature: float,
        rate_limiter: ANRateLimiter | None,
        max_retries: int,
        process_response: Callable[
            [str | tuple[str | None, str | None], BatchItem], BatchResult | None
        ],
        max_new_tokens: int,
    ):
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_retries=max_retries,
            process_response=process_response,
            max_new_tokens=max_new_tokens,
        )

        # assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.rate_limiter = rate_limiter

    @staticmethod
    def is_model_supported(model_id: str) -> bool:
        supported_models = [
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3-opus",
            "claude-3.5-sonnet",
            "claude-3.6-sonnet",
            "claude-3.5-haiku",
            "claude-3.7-sonnet",
        ]
        return any(
            model_id.split("/")[-1].startswith(model) for model in supported_models
        )

    async def process_batch(
        self, items: list[tuple[BatchItem, str]]
    ) -> list[tuple[BatchItem, BatchResult | None]]:
        """Process a batch of items with their corresponding prompts.

        Args:
            items: List of tuples containing (item, prompt)

        Returns:
            List of tuples containing (item, result)
        """
        if len(items) == 0:
            return []

        if self.rate_limiter is None:
            limits = get_anthropic_limits()
            self.rate_limiter = ANRateLimiter(
                requests_per_interval=limits.requests_limit,
                tokens_per_interval=limits.tokens_limit,
                interval_seconds=60,
            )

        async def process_single(
            item: BatchItem, prompt: str
        ) -> tuple[BatchItem, BatchResult | None]:
            result = await generate_an_response_async(
                prompt=prompt,
                model_id=self.model_id,
                client=self.client,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_retries=self.max_retries,
                get_result_from_response=lambda response: self.process_response(
                    response, item
                ),
                rate_limiter=self.rate_limiter,
            )
            return (item, result)

        try:
            tasks = [process_single(item, prompt) for item, prompt in items]
            return await tqdm.gather(*tasks)
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            raise e
        finally:
            await self.client.close()

class ANBatchProcessorWithText(BatchProcessor[BatchItem, BatchResult]):
    def __init__(
        self,
        model_id: str,
        temperature: float,
        rate_limiter: ANRateLimiter | None,
        max_retries: int,
        process_response: Callable[
            [str | tuple[str | None, str | None], BatchItem], BatchResult | None
        ],
        max_new_tokens: int,
    ):
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_retries=max_retries,
            process_response=process_response,
            max_new_tokens=max_new_tokens,
        )

        # assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.rate_limiter = rate_limiter

    @staticmethod
    def is_model_supported(model_id: str) -> bool:
        supported_models = [
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3-opus",
            "claude-3.5-sonnet",
            "claude-3.6-sonnet",
            "claude-3.5-haiku",
            "claude-3.7-sonnet",
        ]
        return any(
            model_id.split("/")[-1].startswith(model) for model in supported_models
        )

    async def process_batch(
        self, items: list[tuple[BatchItem, str]]
    ) -> list[tuple[BatchItem, BatchResult | None]]:
        """Process a batch of items with their corresponding prompts.

        Args:
            items: List of tuples containing (item, prompt)

        Returns:
            List of tuples containing (item, result)
        """
        if len(items) == 0:
            return []

        if self.rate_limiter is None:
            limits = get_anthropic_limits()
            self.rate_limiter = ANRateLimiter(
                requests_per_interval=limits.requests_limit,
                tokens_per_interval=limits.tokens_limit,
                interval_seconds=60,
            )

        async def process_single(
            item: BatchItem, prompt: str
        ) -> tuple[BatchItem, BatchResult | None]:
            result = await generate_an_response_async_with_text(
                prompt=prompt,
                problem_name=item.name,
                model_id=self.model_id,
                client=self.client,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_retries=self.max_retries,
                get_result_from_response=lambda response: self.process_response(
                    response, item
                ),
                rate_limiter=self.rate_limiter,
            )
            return (item, result)

        try:
            tasks = [process_single(item, prompt) for item, prompt in items]
            return await tqdm.gather(*tasks)
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            raise e
        finally:
            await self.client.close()

class ANBatchProcessorWithImage(BatchProcessor[BatchItem, BatchResult]):
    def __init__(
        self,
        model_id: str,
        temperature: float,
        rate_limiter: ANRateLimiter | None,
        max_retries: int,
        process_response: Callable[
            [str | tuple[str | None, str | None], BatchItem], BatchResult | None
        ],
        max_new_tokens: int,
        track_api_usage: str,
    ):
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_retries=max_retries,
            process_response=process_response,
            max_new_tokens=max_new_tokens,
        )

        # assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.rate_limiter = rate_limiter

        self.log_path = setup_logging(True, track_api_usage)

    @staticmethod
    def is_model_supported(model_id: str) -> bool:
        supported_models = [
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3-opus",
            "claude-3.5-sonnet",
            "claude-3.6-sonnet",
            "claude-3.5-haiku",
            "claude-3.7-sonnet",
        ]
        return any(
            model_id.split("/")[-1].startswith(model) for model in supported_models
        )

    async def process_batch(
        self, items: list[tuple[BatchItem, str]]
    ) -> list[tuple[BatchItem, BatchResult | None]]:
        """Process a batch of items with their corresponding prompts.

        Args:
            items: List of tuples containing (item, prompt)

        Returns:
            List of tuples containing (item, result)
        """
        if len(items) == 0:
            return []

        if self.rate_limiter is None:
            limits = get_anthropic_limits()
            self.rate_limiter = ANRateLimiter(
                requests_per_interval=limits.requests_limit,
                tokens_per_interval=limits.tokens_limit,
                interval_seconds=60,
            )

        async def process_single(
            item: BatchItem, prompt: str
        ) -> tuple[BatchItem, BatchResult | None]:
            
            result = await generate_an_response_async_with_image(
                prompt=prompt,
                image_path=item.image_path,
                problem_name=item.name,
                model_id=self.model_id,
                client=self.client,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_retries=self.max_retries,
                get_result_from_response=lambda response: self.process_response(
                    response, item
                ),
                rate_limiter=self.rate_limiter,
            )
            return (item, result)

        try:
            tasks = [process_single(item, prompt) for item, prompt in items]
            return await tqdm.gather(*tasks)
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            raise e
        finally:
            await self.client.close()

def submit_anthropic_batch(
    prompt_by_qrid: dict[QuestionResponseId, str],
    instr_id: str,
    ds_params: DatasetParams,
    evaluated_model_id: str,  # The model being evaluated/generating
    evaluated_sampling_params: SamplingParams,
    evaluator_model_id: str | None = None,  # The model doing the evaluation (if any)
    evaluator_sampling_params: SamplingParams | None = None,
) -> AnthropicBatchInfo:
    """Submit a batch of prompts to Anthropic's batch API.

    Args:
        prompts: List of tuples containing (item, prompt)
        model_id: ID of the model being evaluated/generating
        instr_id: Instruction ID
        ds_params: Dataset parameters
        sampling_params: Sampling parameters for the evaluator
        evaluator_model_id: ID of the model doing the evaluation (if any)
        evaluated_sampling_params: Sampling parameters for the evaluated model (if any)

    Returns:
        Information about the submitted batch
    """
    # For evaluation, evaluator_model_id is the one making API calls
    api_model_id = evaluator_model_id or evaluated_model_id

    is_thinking_model = is_anthropic_thinking_model(api_model_id)
    thinking_budget_tokens = (
        get_budget_tokens(api_model_id) if is_thinking_model else None
    )

    api_model_id = api_model_id.split("/")[-1].split("_")[0]
    api_model_id = ANTHROPIC_MODEL_ALIASES[api_model_id]
    api_sampling_params = evaluator_sampling_params or evaluated_sampling_params

    # Create custom ID mapping and format requests
    custom_id_map = {}
    requests = []
    for i, (q_resp_id, prompt) in enumerate(prompt_by_qrid.items()):
        custom_id = f"{q_resp_id.qid[:31]}__{q_resp_id.uuid[:31]}"
        custom_id_map[custom_id] = q_resp_id

        request_params = MessageCreateParamsNonStreaming(
            model=api_model_id,
            max_tokens=api_sampling_params.max_new_tokens,
            temperature=api_sampling_params.temperature,
            top_p=api_sampling_params.top_p,
            messages=[{"role": "user", "content": prompt}],
        )
        if is_thinking_model:
            assert thinking_budget_tokens is not None
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }
            # Temperature can only be set to 1 for thinking models
            request_params["temperature"] = 1
            # Top-p must be unset for thinking models
            del request_params["top_p"]
            # `max_tokens` must be greater than `thinking.budget_tokens`
            request_params["max_tokens"] = (
                api_sampling_params.max_new_tokens + thinking_budget_tokens
            )

        requests.append(
            Request(
                custom_id=custom_id,
                params=request_params,
            )
        )

    # Submit batch
    message_batch = Anthropic().messages.batches.create(requests=requests)

    # Create batch info
    batch_info = AnthropicBatchInfo(
        batch_id=message_batch.id,
        instr_id=instr_id,
        ds_params=ds_params,
        created_at=datetime.now(timezone.utc).isoformat(),
        custom_id_map=custom_id_map,
        evaluated_model_id=evaluated_model_id,
        evaluated_sampling_params=evaluated_sampling_params,
        evaluator_model_id=evaluator_model_id,
        evaluator_sampling_params=evaluator_sampling_params,
        metadata={}
    )
    batch_info.save()
    return batch_info


def process_batch_results(
    batch_info: AnthropicBatchInfo,
) -> list[tuple[QuestionResponseId, str]]:
    """Process results from a completed batch.

    Args:
        batch_info: Information about the batch

    Returns:
        List of tuples containing (item, response)
        response is None if there was an error processing the response
    """
    client = Anthropic()
    # Get batch status
    message_batch = client.messages.batches.retrieve(batch_info.batch_id)
    if message_batch.processing_status != "ended":
        raise ValueError("Batch still processing")

    processed_results = []

    # Stream results file in memory-efficient chunks, processing one at a time
    for result in client.messages.batches.results(batch_info.batch_id):
        match result.result.type:
            case "succeeded":
                custom_id = result.custom_id
                q_resp_id = batch_info.custom_id_map[custom_id]
                message = result.result.message
                content = message.content
                if not content or len(content) == 0:
                    logging.error(f"Invalid content in request {custom_id}")
                    continue
                if len(content) == 1 and content[0].type == "text":
                    processed_results.append((q_resp_id, content[0].text))
                elif (
                    len(content) == 2
                    and content[0].type == "thinking"
                    and content[1].type == "text"
                ):
                    response = f"<think>{content[0].thinking}</think>{content[1].text}"
                    processed_results.append((q_resp_id, response))
                else:
                    logging.error(f"Invalid content type in request {custom_id}")
                    continue
            case "errored":
                print(f"Error in request {result.custom_id}:\n{result.result.error}")
            case "expired":
                print(f"Request expired {result.custom_id}")

    return processed_results


def cancel_batch(batch_id: str) -> None:
    """Cancel a pending Anthropic batch.

    Args:
        batch_id: The ID of the batch to cancel
    """
    client = Anthropic()
    try:
        client.messages.batches.cancel(batch_id)
    except Exception as e:
        logging.error(f"Error cancelling batch {batch_id}: {str(e)}")
        raise
    finally:
        client.close()

# Add these new classes for direct replacement of OpenRouter
class AnthropicRateLimiter(ANRateLimiter):
    """Anthropic Rate Limiter that can be used as a drop-in replacement for ORRateLimiter.
    
    This inherits from ANRateLimiter but adapts the interface to match ORRateLimiter.
    """
    def __init__(
        self,
        requests_per_interval: int,
        interval_seconds: int,
    ):
        # Get Anthropic limits to determine tokens per interval
        try:
            limits = get_anthropic_limits()
            tokens_per_interval = limits.tokens_limit
        except Exception as e:
            logging.warning(f"Failed to get Anthropic limits: {e}. Using default value.")
            tokens_per_interval = 100000  # Default fallback value
            
        super().__init__(
            requests_per_interval=requests_per_interval,
            tokens_per_interval=tokens_per_interval,
            interval_seconds=interval_seconds,
        )


class AnthropicBatchProcessor(Generic[T, U]):
    """Anthropic Batch Processor that can be used as a drop-in replacement for ORBatchProcessor.
    
    This adapts ANBatchProcessor to match the ORBatchProcessor interface.
    """
    def __init__(
        self,
        model_id: str,
        temperature: float,
        max_new_tokens: int,
        rate_limiter: AnthropicRateLimiter | None = None,
        max_retries: int = 3,
        process_response: Callable[[str | tuple[str | None, str | None], T], U | None] = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        # Ensure process_response is set, even if to a default passthrough
        self.process_response = process_response if process_response else lambda resp, item: resp

        # Initialize Anthropic client
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        
    async def process_batch(self, items: list[tuple[T, str]]) -> list[tuple[T, U | None]]:
        """Process a batch of items with their corresponding prompts.

        Args:
            items: List of tuples containing (item, prompt)

        Returns:
            List of tuples containing (item, result)
        """
        if len(items) == 0:
            return []

        if self.rate_limiter is None:
            limits = get_anthropic_limits()
            self.rate_limiter = AnthropicRateLimiter(
                requests_per_interval=limits.requests_limit,
                interval_seconds=120,
            )

        async def process_single(item: T, prompt: str) -> tuple[T, U | None]:
            # The original code was returning result, metrics_data but generate_an_response_async only returns result
            # Assuming metrics_data is not used downstream or was an error in previous version.
            # If metrics_data is needed, generate_an_response_async needs to be updated.
            raw_result = await generate_an_response_async(
                prompt=prompt,
                model_id=self.model_id,
                client=self.client,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_retries=self.max_retries,
                get_result_from_response=lambda response: self.process_response(response, item),
                rate_limiter=self.rate_limiter,
            )
            # The get_result_from_response callback in generate_an_response_async is what ultimately uses self.process_response.
            # So, raw_result here is the output of self.process_response.
            return (item, raw_result)

        try:
            tasks = [process_single(item, prompt) for item, prompt in items]
            return await tqdm.gather(*tasks)
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            raise e
        finally:
            await self.client.close()