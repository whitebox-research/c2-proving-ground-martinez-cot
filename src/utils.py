import logging
import datetime
from pathlib import Path

LOG_PATH = "data/cot_responses/logs"


def setup_logging(verbose: bool, script_name: str) -> str:
    """Set up logging to both console and file.
    
    Args:
        verbose: Whether to log at INFO level (True) or WARNING level (False)
        script_name: Name of the script for the log filename
    
    Returns:
        Path to the log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path(LOG_PATH)
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


def get_token_usage(model_id: str, usage) -> str:
    """Extract token usage from the response.
    
    Args:
        model_id: The model ID used for the request
        response: The response dictionary containing token usage information
    
    Returns:
        Total token usage for the request
    """
    if "gemini" in model_id:
        token_total = usage.total_token_count
        token_prompt = usage.prompt_token_count
        if usage.thoughts_token_count is None: token_thoughts = 0
        else: token_thoughts = usage.thoughts_token_count

        return f"Total token usage: {token_total}, Prompt tokens: {token_prompt}, Thoughts tokens: {token_thoughts}"
    else:
        return "Unknown model type"
