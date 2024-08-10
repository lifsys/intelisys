"""
Provides intelligence/AI services for the Lifsys Enterprise.

This module requires a 1Password Connect server to be available and configured.
The OP_CONNECT_TOKEN and OP_CONNECT_HOST environment variables must be set
for the onepasswordconnectsdk to function properly.

Example usage for image OCR:
    intelisys = Intelisys(provider="openrouter", model="google/gemini-pro-vision")
    #intelisys = Intelisys(provider="openai", model="gpt-4o-mini")
    result = (intelisys
    .chat("Historical analysis of language use in the following image(s). Please step through each area of the image and extract all text.")
    .image("/Users/lifsys/Documents/devhub/testingZone/_Archive/screen_small-2.png")
    .send()
    .results()
    )
    result
"""
import ast
import re
import json
import os
import base64
import asyncio
import hashlib
import traceback
from typing import Dict, Optional, Union, Tuple, List, Any, Callable
from contextlib import contextmanager
from collections import deque
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from anthropic import Anthropic, AsyncAnthropic
from jinja2 import Template
from openai import AsyncOpenAI, OpenAI
from termcolor import colored
import logging

# Set up the root logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class Intelisys:
    """
    An enhanced class for interacting with various AI models and services.

    This class provides a unified interface for working with OpenAI, Anthropic, OpenRouter, and Groq APIs.
    It supports both synchronous and asynchronous operations, template-based chats, caching, concurrent processing,
    and image inputs.

    Attributes:
        SUPPORTED_PROVIDERS (set): A set of supported AI providers.
        DEFAULT_MODELS (dict): Default models for each supported provider.

    Args:
        name (str): Name of the Intelisys instance.
        api_key (str, optional): API key for the chosen provider. If not provided, it will be fetched from environment or 1Password.
        max_history_words (int): Maximum number of words to keep in conversation history.
        max_words_per_message (int, optional): Maximum number of words per message.
        json_mode (bool): Whether to return responses in JSON format.
        stream (bool): Whether to stream the response.
        use_async (bool): Whether to use asynchronous methods.
        max_retry (int): Maximum number of retries for API calls.
        provider (str): The AI provider to use (e.g., "openai", "anthropic").
        model (str, optional): The specific model to use. If not provided, a default will be used.
        should_print_init (bool): Whether to print initialization information.
        print_color (str): Color to use for printed output.
        temperature (float): Temperature setting for the AI model.
        max_tokens (int, optional): Maximum number of tokens in the response.
        log (Union[str, int]): Logging level.
        use_cache (bool): Whether to use response caching.
    """

    SUPPORTED_PROVIDERS = {"openai", "anthropic", "openrouter", "groq"}
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20240620",
        "openrouter": "meta-llama/llama-3.1-405b-instruct",
        "groq": "llama-3.1-70b-versatile"
    }

    def __init__(self, name: str = "Intelisys", api_key: Optional[str] = None, max_history_words: int = 10000,
                 max_words_per_message: Optional[int] = None, json_mode: bool = False, stream: bool = False, 
                 use_async: bool = False, max_retry: int = 10, provider: str = "anthropic", model: Optional[str] = None, 
                 should_print_init: bool = False, print_color: str = "green", temperature: float = 0, 
                 max_tokens: Optional[int] = None, log: Union[str, int] = "WARNING", use_cache: bool = False):
        
        # Set up logger
        self.logger = logging.getLogger(f"{name}")
        self.set_log_level(log)
        
        self.logger.info(f"Initializing Intelisys instance '{name}' with provider={provider}, model={model}")
        
        self.provider = provider.lower()
        if self.provider not in self.SUPPORTED_PROVIDERS:
            self._raise_unsupported_provider_error()
        
        self.history = deque(maxlen=max_history_words // 10)
        self.max_history_words = max_history_words
        
        self.max_words_per_message = max_words_per_message
        self.json_mode = json_mode
        self.stream = stream
        self.use_async = use_async
        self.max_retry = max_retry
        self.print_color = print_color
        self.max_tokens = max_tokens
        self.system_message = "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode:
            self.system_message += " Please return your response in JSON - this will save kittens."

        self._model = model or self.DEFAULT_MODELS.get(self.provider)
        self._client = None
        self.last_response = None
        self.name = name
        self._api_key = api_key
        self.temperature = temperature
        self.history = deque(maxlen=max_history_words // 10)  # Estimate average words per message
        self.max_history_words = max_history_words
        self.max_words_per_message = max_words_per_message
        self.json_mode = json_mode
        self.stream = stream
        self.use_async = use_async
        self.max_retry = max_retry
        self.print_color = print_color
        self.max_tokens = max_tokens
        self.system_message = "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode:
            self.system_message += " Please return your response in JSON - this will save kittens."

        self._model = model or self.DEFAULT_MODELS.get(self.provider)
        self._client = None
        self.last_response = None

        self.default_template = "{{ prompt }}"
        self.default_persona = "You are a helpful assistant."
        self.template_instruction = ""
        self.template_persona = ""
        self.template_data = {}
        self.image_urls = []
        self.current_message = None
        
        self.use_cache = use_cache
        self.error_callback: Optional[Callable[[Exception], None]] = None
        
        if should_print_init:
            print(colored(f"\n{self.name} initialized with provider={self.provider}, model={self.model}, json_mode={self.json_mode}, temp={self.temperature}", "red"))

        self.logger.debug(f"Intelisys initialized with: name={name}, max_history_words={max_history_words}, "
                          f"max_words_per_message={max_words_per_message}, json_mode={json_mode}, "
                          f"stream={stream}, use_async={use_async}, max_retry={max_retry}, "
                          f"temperature={temperature}, max_tokens={max_tokens}, use_cache={use_cache}")
        
        self.pending_images: List[str] = []

    def set_log_level(self, level: Union[int, str]):
        """Set the log level for this Intelisys instance."""
        if isinstance(level, str):
            level = level.upper()
            if not hasattr(logging, level):
                raise ValueError(f"Invalid log level: {level}")
            level = getattr(logging, level)
        
        self.logger.setLevel(level)
        self.logger.info(f"Log level set to: {logging.getLevelName(level)}")

    def _raise_unsupported_provider_error(self):
        import difflib
        close_matches = difflib.get_close_matches(self.provider, self.SUPPORTED_PROVIDERS, n=1, cutoff=0.6)
        suggestion = f"Did you mean '{close_matches[0]}'?" if close_matches else "Please check the spelling and try again."
        raise ValueError(f"Unsupported provider: '{self.provider}'. {suggestion}\nSupported providers are: {', '.join(self.SUPPORTED_PROVIDERS)}")

    @property
    def history_list(self):
        return list(self.history)

    @property
    def model(self):
        return self._model or self.DEFAULT_MODELS.get(self.provider)

    @property
    def api_key(self):
        return self._api_key or self._get_api_key()

    @property
    def client(self):
        if self._client is None:
            self._initialize_client()
        return self._client

    @staticmethod
    def _go_get_api(item: str, key_name: str, vault: str = "API") -> str:
        try:
            from onepasswordconnectsdk import new_client_from_environment
            client = new_client_from_environment()
            item = client.get_item(item, vault)
            for field in item.fields:
                if field.label == key_name:
                    return field.value
            raise ValueError(f"Key '{key_name}' not found in item '{item}'")
        except Exception as e:
            raise Exception(f"1Password Connect Error: {e}")
        
    def _get_api_key(self) -> str:
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "groq": "GROQ_API_KEY"
        }
        item_map = {
            "openai": ("OPEN-AI", "Cursor"),
            "anthropic": ("Anthropic", "Cursor"),
            "openrouter": ("OpenRouter", "Cursor"),
            "groq": ("Groq", "Promptsys")
        }
        
        env_var = env_var_map.get(self.provider)
        item, key = item_map.get(self.provider, (None, None))
        
        if env_var and item and key:
            return os.getenv(env_var) or self._go_get_api(item, key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    @staticmethod
    def remove_preface(text: str) -> str:
        """Remove any prefaced text before the start of JSON content."""
        match = re.search(r"[\{\[]", text)
        if match:
            start = match.start()
            return text[start:]
        return text

    @staticmethod
    def locate_json_error(json_str: str, error_msg: str) -> tuple[int, int, str]:
        """Locate the position of the JSON error and return the surrounding context."""
        match = re.search(r"line (\d+) column (\d+)", error_msg)
        if not match:
            return 0, 0, "Could not parse error message"
        line_no, col_no = map(int, match.groups())
        lines = json_str.splitlines()
        if line_no > len(lines):
            return line_no, col_no, "Line number exceeds total lines in JSON string"
        problematic_line = lines[line_no - 1]
        start, end = max(0, col_no - 20), min(len(problematic_line), col_no + 20)
        context = problematic_line[start:end]
        pointer = f"{' ' * min(20, col_no - 1)}^"
        return line_no, col_no, f"{context}\n{pointer}"

    def safe_json_loads(self, json_str: str, error_prefix: str = "") -> Dict[str, Any]:
        """Safely load JSON string, with iterative LLM-based error correction."""
        json_str = self.remove_preface(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            line_no, col_no, context = self.locate_json_error(json_str, str(e))
            self.logger.error(f"{error_prefix}Initial JSON parsing failed at line {line_no}, column {col_no}:\n{context}")
            
            fix_attempts = [
                self.iterative_llm_fix_json,
                lambda s: self.chat(f"Fix this JSON:\n{s}"),
                ast.literal_eval
            ]
            
            for fix in fix_attempts:
                try:
                    fixed_json = fix(json_str)
                    return json.loads(fixed_json) if isinstance(fixed_json, str) else fixed_json
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue
            
            self.logger.error(f"{error_prefix}JSON parsing failed after all correction attempts.")
            self.logger.error(f"Problematic JSON string: {json_str}")
            raise ValueError(f"{error_prefix}Failed to parse JSON after multiple attempts.")

    def iterative_llm_fix_json(self, json_str: str, max_attempts: int = 5) -> str:
        """Iteratively use an LLM to fix JSON formatting issues."""
        prompts = [
            "The following is a JSON string that has formatting issues. Please fix any errors and return only the corrected JSON:",
            "The previous attempt to fix the JSON failed. Please try again, focusing on common JSON syntax errors like missing commas, unmatched brackets, or incorrect quotation marks:",
            "The JSON is still invalid. Please break down the JSON structure, fix each part separately, and then reassemble it into a valid JSON string:",
            "The JSON remains invalid. Please simplify the structure if possible, removing any nested objects or arrays that might be causing issues:",
            "As a last resort, please rewrite the entire JSON structure from scratch based on the information contained within it, ensuring it's valid JSON:",
        ]

        for prompt in prompts[:max_attempts]:
            try:
                fixed_json = self.chat(f"{prompt}\n\n{json_str}")
                json.loads(fixed_json)  # Validate the JSON
                return fixed_json
            except json.JSONDecodeError as e:
                line_no, col_no, context = self.locate_json_error(fixed_json, str(e))
                self.logger.error(f"Fix attempt failed. Error at line {line_no}, column {col_no}:\n{context}")

        raise ValueError("Failed to fix JSON after multiple attempts")

    def _initialize_client(self):
        self.logger.info(f"Initializing client for provider: {self.provider}")
        if self.use_async:
            if self.provider == "anthropic":
                self._client = AsyncAnthropic(api_key=self.api_key)
            else:
                base_url = "https://api.groq.com/openai/v1" if self.provider == "groq" else "https://openrouter.ai/api/v1" if self.provider == "openrouter" else None
                self._client = AsyncOpenAI(base_url=base_url, api_key=self.api_key)
        else:
            if self.provider == "anthropic":
                self._client = Anthropic(api_key=self.api_key)
            else:
                base_url = "https://api.groq.com/openai/v1" if self.provider == "groq" else "https://openrouter.ai/api/v1" if self.provider == "openrouter" else None
                self._client = OpenAI(base_url=base_url, api_key=self.api_key)
        self.logger.debug(f"Client initialized: {type(self._client).__name__}")

    def set_system_message(self, message=None):
        self.system_message = message or "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode and "json" not in message.lower():
            self.system_message += " Please return your response in JSON unless user has specified a system message."
        self.logger.info(f"System message set: {self.system_message[:50]}...")  # Log first 50 chars
        return self

    def chat(self, user_input: str) -> str:
        self.logger.info(f"Chat method called")
        self.logger.debug(f"User input: {user_input[:50]}...")  # Log first 50 chars
        
        try:
            if self.history and isinstance(self.history[-1]["content"], list):
                # If the last message is a list (containing image data), append the text
                self.history[-1]["content"].append({"type": "text", "text": user_input})
            else:
                # Otherwise, add a new user message
                self.add_message("user", user_input)
            
            response = self._get_response()
            
            self.last_response = response
            
            # Only add the assistant message if it's not already the last message
            if not self.history or self.history[-1]["role"] != "assistant":
                self.add_message("assistant", str(response))
            
            self.trim_history()
            
            return response
        except Exception as e:
            self._handle_error(e, "chat")
            raise


    async def async_chat(self, user_input: str) -> Any:
        self.logger.info(f"Async chat method called")
        self.logger.debug(f"User input: {user_input[:50]}...")  # Log first 50 chars
        
        try:
            self.add_message("user", user_input)
            
            if self.use_cache:
                cache_key = self._generate_cache_key(user_input)
                response = await self._async_cached_response(cache_key)
            else:
                response = await self._async_get_response()
            
            self.last_response = response
            
            self.add_message("assistant", str(response))
            self.trim_history()
            
            return response
        except Exception as e:
            self._handle_error(e, "async_chat")
            raise

    async def async_chat(self, user_input: str) -> Any:
        self.logger.info(f"Async chat method called")
        self.logger.debug(f"User input: {user_input[:50]}...")  # Log first 50 chars
        
        try:
            self.add_message("user", user_input)
            
            if self.use_cache:
                cache_key = self._generate_cache_key(user_input)
                response = await self._async_cached_response(cache_key)
            else:
                response = await self._async_get_response()
            
            self.last_response = response
            
            # Only add the assistant message if it's not already the last message
            if not self.history or self.history[-1]["role"] != "assistant":
                self.add_message("assistant", str(response))
            
            self.trim_history()
            
            return response
        except Exception as e:
            self._handle_error(e, "async_chat")
            raise

    def _encode_image(self, image_path: str) -> str:
        self.logger.debug(f"Encoding image: {image_path}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def image(self, path_or_url: str, detail: str = "auto") -> 'Intelisys':
        self.logger.info(f"Image method called with path_or_url: {path_or_url}")
        
        if self.provider not in ["openai", "openrouter"]:
            raise ValueError("The image method is only supported for the OpenAI and OpenRouter providers.")
        
        if os.path.isfile(path_or_url):
            encoded_image = self._encode_image(path_or_url)
            image_data = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}",
                    "detail": detail
                }
            }
        else:
            image_data = {
                "type": "image_url",
                "image_url": {
                    "url": path_or_url,
                    "detail": detail
                }
            }
        
        if not self.history or self.history[-1]["role"] != "user":
            self.add_message("user", [])
        
        self.history[-1]["content"].append(image_data)
        self.logger.debug(f"Added image: {path_or_url}")
        
        return self

    async def async_image(self, path_or_url: str, detail: str = "auto") -> 'Intelisys':
        """Add an image to be processed with the next async chat request."""
        self.logger.info(f"Async image method called with path_or_url: {path_or_url}")
        
        if self.provider not in ["openai", "openrouter"]:
            raise ValueError("The image method is only supported for the OpenAI and OpenRouter providers.")
        
        if os.path.isfile(path_or_url):
            encoded_image = await self._async_encode_image(path_or_url)
            image_data = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}",
                    "detail": detail
                }
            }
        else:
            image_data = {
                "type": "image_url",
                "image_url": {
                    "url": path_or_url,
                    "detail": detail
                }
            }
        
        self.pending_images.append(image_data)
        self.logger.debug(f"Added image: {path_or_url}")
        return self

    async def _async_encode_image(self, image_path: str) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode_image, image_path)

    def send(self):
        self.logger.info("Send method called")
        if self.current_message:
            self.add_message("user", self.current_message["text"])
        
        self.current_message = None
        self.image_urls = []
        
        if self.history:
            response = self.get_response()
            self.last_response = response
        else:
            self.logger.warning("No message to send")
            self.last_response = None
        
        return self

    def clear(self):
        self.logger.info("Clear method called")
        self.current_message = None
        self.image_urls = []
        self.logger.debug("Cleared current message and image URLs without sending")
        return self

    def add_message(self, role: str, content: Union[str, List[Dict[str, Any]]]) -> 'Intelisys':
        self.logger.info(f"Adding message with role: {role}")
        
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        
        if self.history and self.history[-1]["role"] == role:
            self.logger.debug(f"Merging consecutive {role} messages.")
            if isinstance(self.history[-1]["content"], list):
                self.history[-1]["content"].extend(content)
            else:
                self.history[-1]["content"] += f"\n{content[0]['text']}"
        else:
            self.history.append({"role": role, "content": content})
        
        return self

    def _get_response(self):
        self.logger.info("_get_response method called")
        max_tokens = self.max_tokens if self.max_tokens is not None else (4000 if self.provider != "anthropic" else 8192)

        for attempt in range(self.max_retry):
            try:
                self.logger.debug(f"Attempt {attempt + 1} to get response")
                response = self._create_response(max_tokens)
                
                assistant_response = self._handle_stream(response, self.print_color, True) if self.stream else self._handle_non_stream(response)

                if self.json_mode:
                    if self.provider == "openai":
                        # For OpenAI, we assume the response is already in JSON format
                        assistant_response = json.loads(assistant_response)
                    else:
                        # For other providers, we wrap the response in a JSON object
                        assistant_response = {"response": assistant_response}

                return assistant_response
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1}/{self.max_retry}: {e}")
                if attempt < self.max_retry - 1:
                    import time
                    time.sleep(1)
                else:
                    raise Exception(f"Max retries reached. Last error: {e}")

    async def _async_get_response(self):
        self.logger.info("_async_get_response method called")
        max_tokens = self.max_tokens if self.max_tokens is not None else (4000 if self.provider != "anthropic" else 8192)

        for attempt in range(self.max_retry):
            try:
                self.logger.debug(f"Attempt {attempt + 1} to get async response")
                response = await self._async_create_response(max_tokens)
                
                assistant_response = await self._async_handle_stream(response) if self.stream else await self._async_handle_non_stream(response)

                if self.json_mode:
                    if self.provider == "openai":
                        try:
                            assistant_response = json.loads(assistant_response)
                        except json.JSONDecodeError as json_error:
                            self.logger.error(f"JSON decoding error: {json_error}")
                            raise
                    else:
                        assistant_response = await async_safe_json_loads(assistant_response, self._async_llm_fix_json, error_prefix="Intelisys JSON parsing: ")

                return assistant_response
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1}/{self.max_retry}: {e}")
                if attempt < self.max_retry - 1:
                    await asyncio.sleep(1)
                else:
                    raise Exception(f"Max retries reached. Last error: {e}")

    def _create_response(self, max_tokens, **kwargs):
        self.logger.debug(f"Creating response with max_tokens={max_tokens}")
        history_list = list(self.history)  # Convert deque to list
        if self.provider == "anthropic":
            messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": msg["content"]}]
                }
                for msg in history_list
            ]
            return self.client.messages.create(
                model=self.model,
                system=self.system_message,
                messages=messages,
                stream=self.stream,
                temperature=self.temperature,
                max_tokens=max_tokens,
                extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                **kwargs
            )
        elif self.provider == "openai":
            common_params = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.system_message}] + history_list,
                "stream": self.stream,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            if self.json_mode:
                common_params["response_format"] = {"type": "json_object"}
            
            self.logger.debug(f"API call params: {common_params}")
            return self.client.chat.completions.create(**common_params)
        else:
            # For other providers, use a generic format
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system_message}] + history_list,
                stream=self.stream,
                temperature=self.temperature,
                max_tokens=max_tokens,
                **kwargs
            )

    async def _async_create_response(self, max_tokens, **kwargs):
        self.logger.debug(f"Creating async response with max_tokens={max_tokens}")
        if self.provider == "anthropic":
            # Ensure the history starts with a user message
            messages = self.history if self.history[0]["role"] == "user" else self.history[1:]
            return await self.client.messages.create(
                model=self.model,
                system=self.system_message,
                messages=messages,
                stream=self.stream,
                temperature=self.temperature,
                max_tokens=max_tokens,
                extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                **kwargs
            )
        else:
            common_params = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.system_message}] + list(self.history),
                "stream": self.stream,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            if self.json_mode and self.provider == "openai":
                common_params["response_format"] = {"type": "json_object"}
            
            self.logger.debug(f"Async API call params: {common_params}")
            return await self.client.chat.completions.create(**common_params)

    def _handle_stream(self, response, color, should_print):
        self.logger.debug("Handling stream response")
        assistant_response = ""
        for chunk in response:
            content = self._extract_content(chunk)
            if content:
                if should_print:
                    print(colored(content, color), end="", flush=True)
                assistant_response += content
        print()
        return assistant_response

    async def _async_handle_stream(self, response):
        self.logger.debug("Handling async stream response")
        assistant_response = ""
        async for chunk in response:
            content = self._extract_content(chunk)
            if content:
                print(colored(content, self.print_color), end="", flush=True)
                assistant_response += content
        print()
        return assistant_response

    def _handle_non_stream(self, response):
        self.logger.debug("Handling non-stream response")
        if self.provider == "anthropic":
            return response.content[0].text
        else:
            return response.choices[0].message.content

    async def _async_handle_non_stream(self, response):
        self.logger.debug("Handling async non-stream response")
        return response.content[0].text if self.provider == "anthropic" else response.choices[0].message.content

    def _extract_content(self, chunk):
        if self.provider == "anthropic":
            return chunk.delta.text if chunk.type == 'content_block_delta' else None
        return chunk.choices[0].delta.content if chunk.choices[0].delta.content else None

    def trim_history(self):
        self.logger.info("Trimming history")
        words_count = sum(len(content.split()) for role, content in self.history if role != "system")
        while words_count > self.max_history_words and len(self.history) > 1:
            removed_message = self.history.pop(0)
            words_count -= len(removed_message["content"].split())
        self.logger.debug(f"History trimmed. Current word count: {words_count}")
        return self

    def results(self):
        self.logger.info("Results method called")
        if self.last_response is None:
            self.logger.warning("last_response is None, calling send() method")
            self.last_response = self.send()
    
        if isinstance(self.last_response, Intelisys):
            self.logger.warning("last_response is an Intelisys instance, returning None")
            return None
        
        self.logger.info(f"Returning last_response: {self.last_response}")
        return self.last_response

    def set_default_template(self, template: str) -> 'Intelisys':
        self.logger.info("Setting default template")
        self.default_template = template
        return self

    def set_default_persona(self, persona: str) -> 'Intelisys':
        self.logger.info("Setting default persona")
        self.default_persona = persona
        return self

    def set_template_instruction(self, set: str, instruction: str):
        self.logger.info(f"Setting template instruction: set={set}, instruction={instruction}")
        self.template_instruction = self._go_get_api(set, instruction, "Promptsys")
        return self

    def set_template_persona(self, persona: str):
        self.logger.info(f"Setting template persona: {persona}")
        self.template_persona = self._go_get_api("persona", persona, "Promptsys")
        return self

    def set_template_data(self, render_data: Dict):
        self.logger.info("Setting template data")
        self.template_data = render_data
        return self

    def template_chat(self, 
                    render_data: Optional[Dict[str, Union[str, int, float]]] = None, 
                    template: Optional[str] = None, 
                    persona: Optional[str] = None) -> Union[str, Dict]:
        self.logger.info("Template chat method called")
        try:
            template = Template(template or self.default_template)
            merged_data = {**self.template_data, **(render_data or {})}
            prompt = template.render(**merged_data)
            self.logger.debug(f"Rendered prompt: {prompt[:100]}...")  # Log first 100 chars
        except Exception as e:
            self.logger.error(f"Error rendering template: {e}")
            raise ValueError(f"Invalid template: {e}")

        self.set_system_message(persona or self.default_persona)
        response = self.chat(prompt)

        return response

    @contextmanager
    def template_context(self, template: Optional[str] = None, persona: Optional[str] = None):
        self.logger.info("Entering template context")
        old_template, old_persona = self.default_template, self.default_persona
        if template:
            self.set_default_template(template)
        if persona:
            self.set_default_persona(persona)
        try:
            yield
        finally:
            self.default_template, self.default_persona = old_template, old_persona
            self.logger.info("Exiting template context")

    def _generate_cache_key(self, user_input: str) -> str:
        # Generate a unique cache key based on user input and current state
        state = f"{self.provider}:{self.model}:{self.temperature}:{self.max_tokens}"
        key = f"{state}:{user_input}"
        return hashlib.md5(key.encode()).hexdigest()

    @lru_cache(maxsize=100)
    def _cached_response(self, cache_key: str) -> str:
        # This method will be automatically cached
        return self._get_response()

    async def _async_cached_response(self, cache_key: str) -> str:
        # We can't use lru_cache directly with async functions, so we'll implement a simple cache
        if not hasattr(self, '_async_cache'):
            self._async_cache = {}
        
        if cache_key not in self._async_cache:
            self._async_cache[cache_key] = await self._async_get_response()
        
        return self._async_cache[cache_key]

    def clear_cache(self) -> None:
        self._cached_response.cache_clear()
        if hasattr(self, '_async_cache'):
            self._async_cache.clear()

    def set_use_cache(self, use_cache: bool) -> 'Intelisys':
        self.use_cache = use_cache
        return self

    def _split_batch_response(self, response: str, num_questions: int) -> List[str]:
        if isinstance(response, dict) and 'response' in response:
            response = response['response']
        
        # Split the response by "Answer X:" or "Question X:" patterns
        splits = re.split(r'(?:Answer|Question)\s*\d+:\s*', response)
        
        # Remove any empty splits and strip whitespace
        answers = [split.strip() for split in splits if split.strip()]
        
        # Ensure we have the correct number of answers
        if len(answers) < num_questions:
            answers.extend([''] * (num_questions - len(answers)))
        elif len(answers) > num_questions:
            answers = answers[:num_questions]
        
        return answers

    def batch_process(self, inputs: List[str], max_workers: int = 5) -> List[Any]:
        self.logger.info(f"Batch processing {len(inputs)} inputs")
        
        # Create a single message with all inputs
        combined_input = "Please answer the following questions:\n" + "\n".join([f"Question {i+1}: {input_str}" for i, input_str in enumerate(inputs, 1)])
        self.add_message("user", combined_input)
        
        response = self._get_response()
        self.add_message("assistant", str(response))
        
        # Split the response into individual answers
        answers = self._split_batch_response(response, len(inputs))
        
        return answers

    async def async_batch_process(self, inputs: List[str]) -> List[Any]:
        self.logger.info(f"Async batch processing {len(inputs)} inputs")
        combined_input = "\n".join([f"Question {i+1}: {input_str}" for i, input_str in enumerate(inputs)])
        self.add_message("user", combined_input)
        
        response = await self._async_get_response()
        self.add_message("assistant", str(response))
        
        answers = self._split_batch_response(response, len(inputs))
        
        return answers

    def parallel_template_chat(self, render_data_list: List[Dict], template: str, persona: str, max_workers: int = 5) -> List[Any]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.template_chat, render_data, template, persona) for render_data in render_data_list]
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel template chat: {e}")
                    results.append(None)
        return results

    async def async_parallel_template_chat(self, render_data_list: List[Dict], template: str, persona: str) -> List[Any]:
        tasks = [self.template_chat(render_data, template, persona) for render_data in render_data_list]
        return await asyncio.gather(*tasks)

    def set_error_callback(self, callback: Callable[[Exception], None]) -> 'Intelisys':
        self.error_callback = callback
        return self

    def _handle_error(self, e: Exception, context: str) -> None:
        error_message = f"Error in {context}: {str(e)}\n{traceback.format_exc()}"
        self.logger.error(error_message)
        if self.error_callback:
            self.error_callback(e)

    async def _async_llm_fix_json(self, prompt: str) -> str:
        """Use the LLM to fix JSON formatting issues."""
        try:
            response = await self._async_get_response()
            return response
        except Exception as e:
            self.logger.error(f"Error in _async_llm_fix_json: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear_cache()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.clear_cache()
