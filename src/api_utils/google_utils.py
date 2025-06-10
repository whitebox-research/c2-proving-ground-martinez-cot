import asyncio
import logging
import os
import time
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Generic, TypeVar
from PIL import Image

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from src.api_utils.batch_processor import (BatchItem, BatchProcessor,
                                                  BatchResult)
from src.typing import (AnthropicBatchInfo, DatasetParams,
                               QuestionResponseId, SamplingParams)

load_dotenv()

# Type variables for generic batch processor
T = TypeVar('T')
U = TypeVar('U')


MAX_THINKING_TIMEOUT = 5 * 60  # 5 minutes
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
    log_filename = f"{script_name}_google_{timestamp}.log"
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
class GORateLimiter:
    requests_per_interval: int
    tokens_per_interval: int
    interval_seconds: int
    input_tokens: float = field(init=False)
    output_tokens: float = field(init=False)
    requests: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False)
    # client: Anthropic = field(default_factory=Anthropic)
    # org_tpm_limit: int = 80000
    org_tpm_limit: int = 16000
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
            f"GORateLimiter initialized with {self.requests_per_interval} requests, "
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
        import random  # Add at top of file if not already present

        for attempt in range(max_retries):
            try:
                await self.acquire(prompt, model)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2**attempt) + random.uniform(0, 1)
                logging.warning(
                    f"Rate limit acquisition failed, retrying in {wait_time:.2f}s: {str(e)}"
                )
                await asyncio.sleep(wait_time)


def get_budget_tokens(model_id: str) -> int:
   return 20000  # Default budget tokens google gemini


async def generate_an_response_async_with_image(
    prompt: str,
    item: BatchItem,
    image_path: str,
    model_id: str,
    client: genai.client, # AsyncAnthropic,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    get_result_from_response: Callable[
        [str | tuple[str | None, str | None]], Any | None
    ],
    rate_limiter: GORateLimiter | None = None,
    is_text: bool = False,
) -> Any | None:
    """Generate a response from a Google model.

    Args:
        prompt: The prompt to run on the model
        model_id: The model ID to call
        client: The Google client
        temperature: Temperature parameter for generation
        max_new_tokens: Maximum number of new tokens to generate
        max_retries: Maximum number of retry attempts for each model
        get_result_from_response: Callback that processes the model response and returns
            either a valid result or None if the response should be retried

    Returns:
        Processed result or None if all attempts failed
    """
    logging.info(f"Running prompt:\n{prompt}")

    is_thinking_model = True # is_anthropic_thinking_model(model_id)
    thinking_budget_tokens = get_budget_tokens(model_id) if is_thinking_model else None

    model_id = model_id.split("/")[-1]
    
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
            }

            if is_thinking_model:
                assert thinking_budget_tokens is not None

                # Temperature can only be set to 1 for thinking models
                # create_params["temperature"] = 1
                # `max_tokens` must be greater than `thinking.budget_tokens`
                # create_params["max_tokens"] = max_new_tokens + thinking_budget_tokens
                # Set the timeout explicitly so that Anthropic doesn't complain about
                # this request _potentially_ taking too long
                # create_params["timeout"] = MAX_THINKING_TIMEOUT

                create_params["config"] = GenerateContentConfig(
                    system_instruction=f"{prompt}",
                    temperature=temperature,
                    maxOutputTokens=max_new_tokens + thinking_budget_tokens,
                    thinking_config=ThinkingConfig(include_thoughts=True)
                )
            
            if not is_text:
                image = Image.open(image_path)
                
                create_params["contents"] = [
                    image,
                    # "Solve this math problem step-by-step, reasoning first and then producing an answer.\n\n",
                ]
            else:
                create_params["contents"] = [
                    item.problem,
                ]
                
            # async operation
            start_time = time.perf_counter()
            go_response = await client.aio.models.generate_content(**create_params) # client.messages.create(**create_params)
            end_time = time.perf_counter()

            if rate_limiter:
                logging.info(f"Got usage: {go_response.usage_metadata}")
                rate_limiter.update_token_usage(go_response.usage_metadata.total_token_count)

            if (
                not go_response
                or len(go_response.candidates[0].content.parts) == 0
                or len(go_response.candidates) == 0
            ):
                logging.info("Invalid response content")
                continue


            if len(go_response.candidates) != 0:
                logging.info(f"Received candidate responses: True")

                if len(go_response.candidates[0].content.parts) != 0 and go_response.candidates[0].content.parts[0].thought:
                    logging.info(f"Received thought response: {go_response.candidates[0].content.parts[0].thought}")

                result = get_result_from_response(
                    (
                        go_response.text,
                        go_response.candidates[0].content.parts[0].text,
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


class GOBatchProcessorWithImage(BatchProcessor[BatchItem, BatchResult]):
    def __init__(
        self,
        model_id: str,
        temperature: float,
        rate_limiter: GORateLimiter | None,
        max_retries: int,
        process_response: Callable[
            [str | tuple[str | None, str | None], BatchItem], BatchResult | None
        ],
        max_new_tokens: int,
        track_api_usage: str,
        is_text: bool,
    ):
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_retries=max_retries,
            process_response=process_response,
            max_new_tokens=max_new_tokens,
        )

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.rate_limiter = rate_limiter
        self.log_path = setup_logging(True, track_api_usage)
        self.is_text = is_text


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
            self.rate_limiter = GORateLimiter(
                requests_per_interval=10,
                tokens_per_interval=16000,
                interval_seconds=60,
            )

        async def process_single(
            item: BatchItem, prompt: str
        ) -> tuple[BatchItem, BatchResult | None]:
            result = await generate_an_response_async_with_image(
                prompt=prompt,
                item=item,
                image_path=item.image_path,
                model_id=self.model_id,
                client=self.client,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_retries=self.max_retries,
                get_result_from_response=lambda response: self.process_response(
                    response, item
                ),
                rate_limiter=self.rate_limiter,
                is_text=self.is_text,
            )
            return (item, result)

        try:
            tasks = [process_single(item, prompt) for item, prompt in items]
            return await tqdm.gather(*tasks)
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            raise e
        finally:
            # await self.client.close()
            # all done
            logging.info("done")