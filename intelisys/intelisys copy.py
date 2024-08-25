"""
Provides intelligence/AI services for the Lifsys Enterprise.

This module requires a 1Password Connect server to be available and configured.
The OP_CONNECT_TOKEN and OP_CONNECT_HOST environment variables must be set
for the onepasswordconnectsdk to function properly.

Example usage for image OCR:
    intelisys = Intelisys(provider="openrouter", model="google/gemini-pro-vision")  # Make sure to use a model that supports image processing
    result = (intelisys
        .image("https://mintlify.s3-us-west-1.amazonaws.com/anthropic/images/how-to-prompt-eng.png")
        .chat("Please provide the complete text in the following image(s).")
    )
    result
"""
import re
import ast
import json
import os
import base64
import io
import requests
from typing import Dict, Optional, Union, Tuple, Any, Type
from contextlib import contextmanager
from PIL import Image
from anthropic import Anthropic, AsyncAnthropic
from jinja2 import Template
from openai import AsyncOpenAI, OpenAI
from termcolor import colored
import logging
from pydantic import BaseModel, ValidationError

# Define the log format
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

# Configure the root logger
logging.basicConfig(level=logging.WARNING, datefmt=DATETIME_FORMAT, format=LOG_FORMAT, force=True)

# Create a logger for this module
logger = logging.getLogger("Global")
logger.setLevel(logging.INFO)

def remove_preface(text: str) -> str:
    """Remove any prefaced text before the start of JSON content."""
    match: Optional[re.Match] = re.search(r"[\{\[]", text)
    if match:
        start: int = match.start()
        return text[start:]
    return text

def locate_json_error(json_str: str, error_msg: str) -> Tuple[int, int, str]:
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

def iterative_llm_fix_json(json_string: str, max_attempts: int = 5, intelisys_instance=None) -> str:
    logger.info(f"Starting iterative_llm_fix_json with input: {json_string}")
    if intelisys_instance is None:
        intelisys_instance = Intelisys(provider="openai", model="gpt-3.5-turbo", api_key="dummy_key")
    attempts = 0
    
    while attempts < max_attempts:
        prompt = f"Fix this JSON: {json_string}"
        logger.debug(f"Sending prompt to AI: {prompt}")
        response = intelisys_instance.chat(prompt)
        logger.debug(f"Received response from AI: {response}")
        
        try:
            json.loads(response)  # Try to parse the AI's response
            logger.info(f"Successfully parsed JSON on attempt {attempts + 1}")
            return response  # If successful, return the AI's response
        except json.JSONDecodeError:
            logger.warning(f"JSON parsing failed on attempt {attempts + 1}")
            json_string = response  # Update json_string with the AI's response for the next attempt
        
        attempts += 1
    
    logger.warning(f"Reached max attempts. Returning: {json_string}")
    return json_string  # Return the last attempt even if it's not valid JSON

def safe_json_loads(json_str: str, error_prefix: str = "") -> Dict:
    """Safely convert any string input into JSON, with iterative LLM-based error correction."""
    if json_str is None:
        raise ValueError(f"{error_prefix}Input is None")

    if not isinstance(json_str, str):
        raise TypeError(f"{error_prefix}Input must be a string, not {type(json_str)}")

    json_str = remove_preface(json_str)
    
    fix_attempts = [
        json.loads,
        lambda s: Intelisys(
            provider="openai", 
            model="gpt-4o-mini",
            json_mode=True) \
            .set_system_message("Convert the following text into valid JSON. If it's already valid JSON, return it as is.") \
            .chat(f"Convert this to JSON:\n{s}"),
        iterative_llm_fix_json,
        lambda s: ast.literal_eval(s) if s.strip().startswith('{') else {"content": s}
    ]
    
    for fix in fix_attempts:
        try:
            fixed_json = fix(json_str)
            if isinstance(fixed_json, dict):
                return fixed_json
            elif isinstance(fixed_json, str):
                # If it's still a string, try to parse it as JSON one more time
                return json.loads(fixed_json)
        except Exception as e:
            logger.debug(f"{error_prefix}JSON conversion attempt failed: {str(e)}")
            continue
    
    # If all attempts fail, create a simple JSON object with the original string as content
    logger.warning(f"{error_prefix}Failed to convert to JSON. Creating a simple JSON object.")
    return {"content": json_str}

class Intelisys:
    """
    A class for interacting with various AI providers and models.

    This class provides a unified interface for chatting with AI models,
    handling image inputs, and managing conversation history.

    Attributes:
        SUPPORTED_PROVIDERS (set): Set of supported AI providers.
        DEFAULT_MODELS (dict): Default models for each provider.

    Args:
        name (str): Name of the Intelisys instance.
        api_key (str, optional): API key for the chosen provider.
        max_history_words (int): Maximum number of words to keep in conversation history.
        max_words_per_message (int, optional): Maximum words per message.
        json_mode (bool): Whether to return responses in JSON format.
        stream (bool): Whether to stream the response.
        use_async (bool): Whether to use async methods.
        max_retry (int): Maximum number of retries for API calls.
        provider (str): AI provider to use (e.g., "openai", "anthropic").
        model (str, optional): Specific model to use.
        should_print_init (bool): Whether to print initialization details.
        print_color (str): Color for printed output.
        temperature (float): Temperature for response generation.
        max_tokens (int, optional): Maximum tokens for response.
        log (str or int): Logging level.

    Usage:
        intelisys = Intelisys(provider="openai", model="gpt-4")
        response = intelisys.chat("Hello, how are you?")
    """
    # Define the log format
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
 
    SUPPORTED_PROVIDERS = {"openai", "anthropic", "openrouter", "groq"}
    DEFAULT_MODELS = {
        "openai": "gpt-4o-2024-08-06",
        "anthropic": "claude-3-5-sonnet-20240620",
        "openrouter": "meta-llama/llama-3.1-405b-instruct",
        "groq": "llama-3.1-8b-instant"
    }

    def __init__(self, name="Intelisys", api_key=None, max_history_words=0,
                 max_words_per_message=None, json_mode=False, stream=False, use_async=False,
                 max_retry=10, provider="anthropic", model=None, should_print_init=False,
                 print_color="green", temperature=0, max_tokens=None, log: Union[str, int] = "WARNING"):
        """
        Initialize the Intelisys instance.

        Args:
            name (str): Name of the Intelisys instance.
            api_key (str, optional): API key for the chosen provider.
            max_history_words (int): Maximum number of words to keep in conversation history.
            max_words_per_message (int, optional): Maximum words per message.
            json_mode (bool): Whether to return responses in JSON format.
            stream (bool): Whether to stream the response.
            use_async (bool): Whether to use async methods.
            max_retry (int): Maximum number of retries for API calls.
            provider (str): AI provider to use (e.g., "openai", "anthropic").
            model (str, optional): Specific model to use.
            should_print_init (bool): Whether to print initialization details.
            print_color (str): Color for printed output.
            temperature (float): Temperature for response generation.
            max_tokens (int, optional): Maximum tokens for response.
            log (str or int): Logging level.
        """
        
        # Set up logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self.set_log_level(log)
        
        self.logger.debug(f"Initializing Intelisys instance '{name}' with provider={provider}, model={model}")
        
        self.provider = provider.lower()
        if self.provider not in self.SUPPORTED_PROVIDERS:
            self._raise_unsupported_provider_error()
        
        self.name = name
        self._api_key = api_key
        self.temperature = temperature
        self.history = []
        self.max_history_words = max_history_words
        self.max_words_per_message = max_words_per_message
        self.json_mode = json_mode
        if self.json_mode and self.provider != "openai":
            self.logger.debug(f"json_mode=True is set for provider '{self.provider}'")
        self.stream = stream
        self.use_async = use_async
        self.max_retry = max_retry
        self.print_color = print_color
        self.max_tokens = max_tokens
        self.system_message = "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode:
            self.system_message += " Please return your response in JSON"

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
        
        if should_print_init:
            print(colored(f"\n{self.name} initialized with provider={self.provider}, model={self.model}, json_mode={self.json_mode}, temp={self.temperature}", "red"))

        self.logger.debug(f"Intelisys initialized with: name={name}, max_history_words={max_history_words}, "
                          f"max_words_per_message={max_words_per_message}, json_mode={json_mode}, "
                          f"stream={stream}, use_async={use_async}, max_retry={max_retry}, "
                          f"temperature={temperature}, max_tokens={max_tokens}")

        self.output_model = None
        self.structured_output = None

    def set_log_level(self, level: Union[int, str]):
        if isinstance(level, str):
            level = level.upper()
            if not hasattr(logging, level):
                raise ValueError(f"Invalid log level: {level}")
            level = getattr(logging, level)
        
        self.logger.setLevel(level)
        
        # Remove all existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add a single handler with the correct formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter(self.LOG_FORMAT, datefmt=self.DATETIME_FORMAT)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Prevent the logger from propagating messages to the root logger
        self.logger.propagate = False
        
        self.logger.debug("Log level set to: %s", logging.getLevelName(level))

    def _raise_unsupported_provider_error(self):
        import difflib
        close_matches = difflib.get_close_matches(self.provider, self.SUPPORTED_PROVIDERS, n=1, cutoff=0.6)
        suggestion = f"Did you mean '{close_matches[0]}'?" if close_matches else "Please check the spelling and try again."
        raise ValueError(f"Unsupported provider: '{self.provider}'. {suggestion}\nSupported providers are: {', '.join(self.SUPPORTED_PROVIDERS)}")

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
        
    def _get_api_key(self):
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

    def _initialize_client(self):
        self.logger.debug(f"Initializing client for provider: {self.provider}")
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
        """
        Set the system message for the conversation.

        Args:
            message (str, optional): The system message to set. If None, a default message is used.

        Returns:
            self: The Intelisys instance for method chaining.

        Usage:
            intelisys.set_system_message("You are a helpful assistant specialized in Python programming.")
        """
        self.system_message = message or "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode and "json" not in message.lower():
            self.system_message += " Please return your response in JSON unless user has specified a system message."
        self.logger.debug(f"System message set: {self.system_message[:50]}...")  # Log first 50 chars
        return self

    def chat(self, user_input):
        """
        Send a chat message to the AI and return the response.

        Args:
            user_input (str): The user's message to send to the AI.

        Returns:
            Union[str, BaseModel]: The AI's response as a string or a Pydantic model instance if structured output is used.

        Usage:
            response = intelisys.chat("What is the capital of France?")
        """
        self.logger.debug("*Chat*")
        self.logger.debug(f"User input: {user_input[:50]}...")
        self.current_message = {"role": "user", "content": user_input}
        if self.max_history_words > 0:
            self.add_message("user", user_input)
        
        try:
            response = self._create_response(self.max_tokens or (4000 if self.provider != "anthropic" else 8192))
            self.logger.debug(f"Raw API response: {response}")
            result = self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error in chat method: {str(e)}")
            raise
        
        self.current_message = None
        self.image_urls = []  # Clear image URLs after sending
        
        return result

    def _encode_image(self, image_path: str) -> str:
        self.logger.debug(f"Encoding image: {image_path}")
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='PNG')
            return base64.b64encode(byte_arr.getvalue()).decode('utf-8')

    def image(self, path_or_url: str, detail: str = "auto"):
        """
        Add an image to the current message for image-based AI tasks.

        Args:
            path_or_url (str): Local file path or URL of the image.
            detail (str, optional): Level of detail for image analysis (default is "auto").

        Returns:
            self: The Intelisys instance for method chaining.

        Raises:
            ValueError: If the provider doesn't support image inputs.
            FileNotFoundError: If the local image file is not found.

        Usage:
            intelisys.chat("Describe this image").image("/path/to/image.jpg").get_response()
        """
        self.logger.debug(f"Image method called with path_or_url: {path_or_url}")
        if self.provider not in ["openai", "openrouter"]:
            raise ValueError("The image method is only supported for the OpenAI and OpenRouter providers.")
        
        if path_or_url.startswith(('http://', 'https://')):
            response = requests.get(path_or_url)
            response.raise_for_status()
            image_data = base64.b64encode(response.content).decode('utf-8')
        else:
            # Validate local file path
            if not os.path.exists(path_or_url):
                raise FileNotFoundError(f"Image file not found: {path_or_url}")
            with open(path_or_url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

        self.image_urls.append(f"data:image/jpeg;base64,{image_data}")
        self.logger.debug(f"Added image: {path_or_url}")
        return self

    def _create_response(self, max_tokens, **kwargs):
        if self.max_history_words > 0:
            messages = self.history.copy()
        else:
            # Use the current_message directly, as it's already properly formatted
            messages = [self.current_message]
        
        common_params = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
            "temperature": self.temperature,
        }

        if self.provider == "anthropic":
            # For Anthropic, adjust max_tokens and add the beta header
            anthropic_max_tokens = min(max_tokens, 4096)  # Ensure it doesn't exceed 4096
            extra_headers = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}
            
            return self.client.messages.create(
                system=self.system_message,
                max_tokens=anthropic_max_tokens,
                extra_headers=extra_headers,
                **common_params,
                **kwargs
            )
        else:
            # For other providers
            if max_tokens:
                common_params["max_tokens"] = max_tokens

            if self.image_urls and self.provider in ["openai", "openrouter"]:
                last_message = common_params["messages"][-1]
                content = []

                if isinstance(last_message["content"], str):
                    content.append({"type": "text", "text": last_message["content"]})

                for image_url in self.image_urls:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })

                last_message["content"] = content

            if self.json_mode and self.provider == "openai":
                common_params["response_format"] = {"type": "json_object"}
            
            if self.system_message:
                common_params["messages"].insert(0, {"role": "system", "content": self.system_message})

            if self.provider == "openai" and self.output_model:
                common_params["response_format"] = {"type": "json_object"}
                common_params["functions"] = [{
                    "name": "output",
                    "parameters": self.output_model.model_json_schema()
                }]
                common_params["function_call"] = {"name": "output"}
                
                # Add JSON instruction to the system message
                json_instruction = "Please return your response in JSON format according to the specified schema."
                if common_params["messages"][0]["role"] == "system":
                    common_params["messages"][0]["content"] += f" {json_instruction}"
                else:
                    common_params["messages"].insert(0, {"role": "system", "content": json_instruction})

            self.logger.debug(f"API call params: {common_params}")
            return self.client.chat.completions.create(**common_params, **kwargs)

    def _handle_response(self, response):
        logger = logging.getLogger("handle_response")
        logger.info("Handling response")
        if self.stream:
            logger.debug("Handling stream response")
            assistant_response = self._handle_stream(response, self.print_color, True)
        else:
            logger.debug("Handling non-stream response")
            assistant_response = self._handle_non_stream(response)

        logger.debug(f"Raw assistant response: {assistant_response}")

        if assistant_response is None:
            raise ValueError("Received None response from assistant")

        if self.json_mode:
            self.logger.debug("JSON mode is enabled, attempting to parse response")
            if self.provider == "openai":
                try:
                    assistant_response = json.loads(assistant_response)
                except json.JSONDecodeError as json_error:
                    self.logger.error(f"OpenAI JSON decoding error: {json_error}")
                    raise
            else:
                try:
                    assistant_response = safe_json_loads(assistant_response, error_prefix="Intelisys JSON parsing: ")
                except Exception as json_error:
                    self.logger.error(f"safe_json_loads error: {json_error}")
                    raise

        if self.provider == "openai" and self.output_model:
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "output":
                try:
                    self.structured_output = self.output_model.model_validate_json(function_call.arguments)
                except ValidationError:
                    self.logger.warning("Failed to validate structured output")
                    self.structured_output = None
            else:
                self.structured_output = None

        self.logger.debug(f"Final processed assistant response: {assistant_response}")
        self.add_message("assistant", str(assistant_response))
        self.trim_history()
        return assistant_response

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

    def _handle_non_stream(self, response):
        self.logger.debug("Handling non-stream response")
        if self.provider == "anthropic":
            return response.content[0].text
        elif self.provider == "openai":
            message = response.choices[0].message
            if message.function_call:
                return message.function_call.arguments
            else:
                return message.content
        else:
            return response.choices[0].message.content

    def _extract_content(self, chunk):
        if self.provider == "anthropic":
            return chunk.delta.text if chunk.type == 'content_block_delta' else None
        return chunk.choices[0].delta.content if chunk.choices[0].delta.content else None

    def trim_history(self):
        if self.max_history_words > 0:
            self.logger.info("Trimming history")
            words_count = sum(len(str(m["content"]).split()) for m in self.history if m["role"] != "system")
            while words_count > self.max_history_words and len(self.history) > 1:
                removed_message = self.history.pop(0)
                words_count -= len(str(removed_message["content"]).split())
            self.logger.debug(f"History trimmed. Current word count: {words_count}")
        else:
            self.history.clear()
            self.logger.debug("History cleared (max_history_words is 0)")
        return self

    def add_message(self, role, content):
        self.logger.debug(f"Adding message with role: {role}")
        self.logger.debug(f"Message content: {content[:50]}...")  # Log first 50 chars
        if role == "user" and self.max_words_per_message:
            if isinstance(content, str):
                content += f" please use {self.max_words_per_message} words or less"
            elif isinstance(content, list) and content and isinstance(content[0], dict) and content[0].get('type') == 'text':
                content[0]['text'] += f" please use {self.max_words_per_message} words or less"

        if self.max_history_words > 0:
            self.history.append({"role": role, "content": content})
            self.trim_history()
        return self

    def set_default_template(self, template: str) -> 'Intelisys':
        self.logger.debug("*Set Default template*")
        self.default_template = template
        return self

    def set_default_persona(self, persona: str) -> 'Intelisys':
        self.logger.debug("*Setting default persona*")
        self.default_persona = persona
        return self

    def set_template_instruction(self, set: str, instruction: str):
        self.logger.debug(f"Setting template instruction: set={set}, instruction={instruction}")
        self.template_instruction = self._go_get_api(set, instruction, "Promptsys")
        return self

    def set_template_persona(self, persona: str):
        self.logger.debug(f"Setting template persona: {persona}")
        self.template_persona = self._go_get_api("persona", persona, "Promptsys")
        return self

    def set_template_data(self, render_data: Dict):
        self.logger.debug("Setting template data")
        self.template_data = render_data
        return self

    def template_chat(self, 
                    render_data: Optional[Dict[str, Union[str, int, float]]] = None, 
                    template: Optional[str] = None, 
                    persona: Optional[str] = None) -> Union[str, BaseModel]:
        """
        Send a chat message using a template and get the AI's response.

        Args:
            render_data (dict, optional): Data to render the template with.
            template (str, optional): The template string to use. If None, uses the default template.
            persona (str, optional): The persona to use for the system message. If None, uses the default persona.

        Returns:
            Union[str, BaseModel]: The AI's response as a string or a Pydantic model instance if structured output is used.

        Raises:
            ValueError: If there's an error rendering the template.

        Usage:
            response = intelisys.template_chat(
                render_data={"name": "Alice", "question": "What's the weather like?"},
                template="Hello {{name}}, {{question}}",
                persona="You are a weather expert."
            )
        """
        self.logger.info("*Template*")
        try:
            template = Template(template or self.default_template)
            merged_data = {**self.template_data, **(render_data or {})}
            prompt = template.render(**merged_data)
            self.logger.debug(f"Rendered prompt: {prompt[:100]}...")  # Log first 100 chars
        except Exception as e:
            self.logger.error(f"Error rendering template: {e}")
            raise ValueError(f"Invalid template: {e}")

        self.set_system_message(persona or self.default_persona)
        return self.chat(prompt)

    def transcript(self, audio_file_path: str, model: str = "whisper-1") -> str:
        """
        Transcribe an audio file using OpenAI's Whisper model.

        Args:
            audio_file_path (str): Path to the audio file to transcribe.
            model (str, optional): The model to use for transcription. Defaults to "whisper-1".

        Returns:
            str: The transcribed text.

        Raises:
            ValueError: If the provider is not OpenAI or if the audio file is not found.

        Usage:
            transcription = intelisys.transcript("/path/to/audio.mp3")
        """
        self.logger.debug(f"Transcribing audio file: {audio_file_path}")

        if self.provider != "openai":
            raise ValueError("The transcript method is only supported for the OpenAI provider.")

        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file
                )
            
            self.logger.debug("Transcription completed successfully")
            return transcript.text
        except Exception as e:
            self.logger.error(f"Error during transcription: {str(e)}")
            raise

    @contextmanager
    def template_context(self, template: Optional[str] = None, persona: Optional[str] = None):
        self.logger.debug("*Template context*")
        old_template, old_persona = self.default_template, self.default_persona
        if template:
            self.set_default_template(template)
        if persona:
            self.set_default_persona(persona)
        try:
            yield
        finally:
            self.default_template, self.default_persona = old_template, old_persona
            self.logger.debug("Exiting template context")

    # Async methods
    async def chat_async(self, user_input, **kwargs):
        self.logger.debug("Async chat method called")
        await self.add_message_async("user", user_input)
        self.last_response = await self.get_response_async(**kwargs)
        return self.last_response

    async def add_message_async(self, role, content):
        self.logger.debug(f"Async adding message with role: {role}")
        self.add_message(role, content)
        return self

    async def set_system_message_async(self, message=None):
        self.logger.debug("Async setting system message")
        self.set_system_message(message)
        return self

    async def get_response_async(self, color=None, should_print=True, **kwargs):
        self.logger.debug("Async get_response method called")
        color = color or self.print_color
        max_tokens = kwargs.pop('max_tokens', 4000 if self.provider != "anthropic" else 8192)

        response = await self._create_response_async(max_tokens, **kwargs)

        assistant_response = await self._handle_stream_async(response, color, should_print) if self.stream else await self._handle_non_stream_async(response)

        if self.json_mode and self.provider == "openai":
            try:
                assistant_response = json.loads(assistant_response)
            except json.JSONDecodeError as json_error:
                self.logger.error(f"JSON decoding error: {json_error}")
                raise

        await self.add_message_async("assistant", str(assistant_response))
        await self.trim_history_async()
        return assistant_response

    async def _create_response_async(self, max_tokens, **kwargs):
        self.logger.debug(f"Creating async response with max_tokens={max_tokens}")
        if self.provider == "anthropic":
            return await self.client.messages.create(
                model=self.model,
                system=self.system_message,
                messages=self.history,
                stream=self.stream,
                temperature=self.temperature,
                max_tokens=max_tokens,
                extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                **kwargs
            )
        else:
            common_params = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.system_message}] + self.history,
                "stream": self.stream,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            if self.json_mode and self.provider == "openai":
                common_params["response_format"] = {"type": "json_object"}
            
            if self.provider == "openai" and self.output_model:
                common_params["response_format"] = {"type": "json_object"}
                common_params["functions"] = [{
                    "name": "output",
                    "parameters": self.output_model.model_json_schema()
                }]
                common_params["function_call"] = {"name": "output"}

            return await self.client.chat.completions.create(**common_params)

    async def _handle_stream_async(self, response, color, should_print):
        self.logger.debug("Handling async stream response")
        assistant_response = ""
        async for chunk in response:
            content = self._extract_content_async(chunk)
            if content:
                if should_print:
                    print(colored(content, color), end="", flush=True)
                assistant_response += content
        print()
        return assistant_response

    async def _handle_non_stream_async(self, response):
        self.logger.debug("Handling async non-stream response")
        return response.content[0].text if self.provider == "anthropic" else response.choices[0].message.content

    def _extract_content_async(self, chunk):
        if self.provider == "anthropic":
            return chunk.delta.text if chunk.type == 'content_block_delta' else None
        return chunk.choices[0].delta.content if chunk.choices[0].delta.content else None

    async def trim_history_async(self):
        self.logger.debug("Async trimming history")
        self.trim_history()
        return self

    def set_output_model(self, model: Type[BaseModel]):
        """
        Set the Pydantic model for structured output (OpenAI provider only).

        Args:
            model (Type[BaseModel]): The Pydantic model defining the structure of the output.

        Returns:
            self: The Intelisys instance for method chaining.
        """
        if self.provider != "openai":
            self.logger.warning("Structured output is only supported for the OpenAI provider. This setting will be ignored.")
        else:
            self.output_model = model
        return self

    def results(self) -> Union[str, BaseModel, None]:
        """
        Get the results of the last chat operation.

        Returns:
            Union[str, BaseModel, None]: The chat response as a string, 
            a Pydantic model instance (if structured output is used with OpenAI), 
            or None if not available.
        """
        if self.provider == "openai" and self.structured_output:
            return self.structured_output
        return self.last_response

    async def template_chat_async(self, 
                                render_data: Optional[Dict[str, Union[str, int, float]]] = None, 
                                template: Optional[str] = None, 
                                persona: Optional[str] = None, 
                                parse_json: bool = False) -> 'Intelisys':
        """
        Asynchronously send a chat message using a template and get the AI's response.

        Args:
            render_data (dict, optional): Data to render the template with.
            template (str, optional): The template string to use. If None, uses the default template.
            persona (str, optional): The persona to use for the system message. If None, uses the default persona.

        Returns:
            Intelisys: The Intelisys instance for method chaining.

        Raises:
            ValueError: If there's an error rendering the template or processing the response.

        Usage:
            response = await intelisys.template_chat_async(
                render_data={"name": "Bob", "question": "What's the capital of France?"},
                template="Hello {{name}}, {{question}}",
                persona="You are a geography expert."
            )
            result = intelisys.last_response
        """
        self.logger.debug("Async template chat method called")
        try:
            template = Template(template or self.default_template)
            merged_data = {**self.template_data, **(render_data or {})}
            prompt = template.render(**merged_data)
            self.logger.debug(f"Rendered prompt: {prompt[:50]}...")  # Log first 50 chars
        except Exception as e:
            self.logger.error(f"Error rendering template: {e}")
            raise ValueError(f"Invalid template: {e}")

        await self.set_system_message_async(persona or self.default_persona)
        response = await self.chat_async(prompt)
        
        if self.json_mode:
            if isinstance(response, dict):
                self.last_response = response
            elif isinstance(response, str):
                try:
                    self.last_response = json.loads(response)
                except json.JSONDecodeError:
                    self.last_response = safe_json_loads(response, error_prefix="Intelisys async template chat JSON parsing: ")
            else:
                self.logger.error(f"Unexpected response type: {type(response)}")
                raise ValueError(f"Unexpected response type: {type(response)}")
        else:
            self.last_response = response
        
        return self
