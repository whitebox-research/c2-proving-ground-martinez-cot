import asyncio
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar
from PIL import Image

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from src.api_utils.batch_processor import (BatchItem, BatchProcessor, BatchResult)
from src.utils import setup_logging, get_token_usage

load_dotenv()

# Type variables for generic batch processor
T = TypeVar('T')
U = TypeVar('U')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



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


async def generate_response_async(
    prompt: str,
    item: BatchItem,
    model_id: str,
    client: genai.client,
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
        get_result_from_response: Callback that processes the model response and returns either a valid result or None

    Returns:
        Processed result or None if all attempts failed
    """


    is_thinking_model = True
    thinking_budget_tokens = get_budget_tokens(model_id) if is_thinking_model else None

    model_id = model_id.split("/")[-1]
    
    if is_thinking_model: logging.info(f"Thinking model with thinking budget of {thinking_budget_tokens} tokens")
    if is_text:
        logging.info(f"Processing {item.name} as text imput")
    else:
        logging.info(f"Processing {item.name} as image input")
 
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(f"Retry attempt {attempt} of {max_retries} for generating a response")

            if rate_limiter:
                await rate_limiter.acquire_with_backoff(prompt, model_id)

            create_params = { "model": model_id }

            if is_thinking_model:
                assert thinking_budget_tokens is not None

                create_params["config"] = GenerateContentConfig(
                    system_instruction=f"{prompt}",
                    temperature=temperature,
                    maxOutputTokens=max_new_tokens + thinking_budget_tokens,
                    thinking_config=ThinkingConfig(includeThoughts=True, thinkingBudget=thinking_budget_tokens)
                )
            
            if not is_text:
                image = Image.open(item.image_path)                
                create_params["contents"] = [ image ]
            else:
                create_params["contents"] = [ item.problem ]
                
            # async operation
            start_time = time.perf_counter()
            go_response = await client.aio.models.generate_content(**create_params)
            end_time = time.perf_counter()

            if (
                not go_response
                or len(go_response.candidates[0].content.parts) == 0
                or len(go_response.candidates) == 0
            ):
                logging.info("Invalid response content")
                continue

            logging.info(f"Got usage: {get_token_usage(model_id, go_response.usage_metadata)}")
            
            if rate_limiter: rate_limiter.update_token_usage(go_response.usage_metadata.total_token_count)

            result = get_result_from_response(
                (
                    None,
                    go_response.candidates[0].content.parts[0].text,
                )
            )

            if result is not None:
                logging.info(f"Found valid result for item {item.name} on attempt {attempt + 1}")
                logging.info(f"Response: {go_response.candidates[0].content.parts[0].text[:100]}...")  # Limiting logging length of response

                return result

            logging.info(f"Invalid result on attempt {attempt + 1} for model {model_id}, retrying...")

            attempt += 1
            continue

        except Exception as e:
            if attempt == max_retries:
                logging.warning(f"Failed to process response after {max_retries} retries for problem {item.problem}")
                return None
            
            logging.warning(f"Error on attempt {attempt + 1} for model {model_id}: {str(e)}, retrying...")
            
            attempt += 1
            continue

    return None


class GOBatchProcessor(BatchProcessor[BatchItem, BatchResult]):
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
        self.is_text = is_text


    async def process_batch(self, items: list[tuple[BatchItem, str]]) -> list[tuple[BatchItem, BatchResult | None]]:
        """Process a batch of items with their corresponding prompts.

        Args:
            items: List of tuples containing (item, prompt)

        Returns:
            List of tuples containing (item, result)
        """
        if len(items) == 0:
            return []

        if self.rate_limiter is None: self.rate_limiter = GORateLimiter(requests_per_interval=10, tokens_per_interval=16000, interval_seconds=60)

        async def process_single(item: BatchItem, prompt: str) -> tuple[BatchItem, BatchResult | None]:
            
            result = await generate_response_async(
                prompt=prompt,
                item=item,
                model_id=self.model_id,
                client=self.client,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_retries=self.max_retries,
                get_result_from_response=lambda response: self.process_response(response, item),
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
        # finally:
        #    logging.info("All done")