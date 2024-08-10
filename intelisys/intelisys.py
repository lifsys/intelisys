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
    .get_response()    )
    result
"""
import re
import ast
import json
import os
import base64
import io
from typing import Dict, Optional, Union, Tuple
from contextlib import contextmanager
from PIL import Image
from anthropic import Anthropic, AsyncAnthropic
from jinja2 import Template
from openai import AsyncOpenAI, OpenAI
from termcolor import colored
import logging

# Set up the root logger
LOG_FORMAT = "%(asctime)s %(levelname)s - %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("main")

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
        response = intelisys.chat("Hello, how are you?").get_response()
    """
    LOG_FORMAT = "%(asctime)s %(levelname)s - %(name)s: %(message)s"

    @classmethod
    def _configure_logger(cls, name: str, level: Union[int, str] = "WARNING"):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove any existing handlers to avoid duplication
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add a new handler with the correct format
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
        logger.addHandler(handler)
        
        return logger
    
    SUPPORTED_PROVIDERS = {"openai", "anthropic", "openrouter", "groq"}
    DEFAULT_MODELS = {
        "openai": "gpt-4o-2024-08-06",
        "anthropic": "claude-3-5-sonnet-20240620",
        "openrouter": "meta-llama/llama-3.1-405b-instruct",
        "groq": "llama-3.1-8b-instant"
    }

    def __init__(self, name="Intelisys", api_key=None, max_history_words=10000,
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
        logging.basicConfig(format=self.LOG_FORMAT)
        self.logger = logging.getLogger("init")
        self.set_log_level(log)
        
        self.logger.info(f"Initializing Intelisys instance '{name}' with provider={provider}, model={model}")
        
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
            self.logger.warning(f"json_mode=True is set for provider '{self.provider}'")
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

    def set_log_level(self, level: Union[int, str] = "WARNING"):
        """
        Set the log level for this Intelisys instance.

        Args:
            level (int or str): The log level to set. Can be a string (e.g., "DEBUG", "INFO")
                                or an integer constant from the logging module.
                                Defaults to "WARNING".

        Raises:
            ValueError: If an invalid log level string is provided.
        """
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
        self.logger.info(f"System message set: {self.system_message[:50]}...")  # Log first 50 chars
        return self

    def chat(self, user_input):
        """
        Send a chat message to the AI and prepare for a response.

        Args:
            user_input (str): The user's message to send to the AI.

        Returns:
            self: The Intelisys instance for method chaining.

        Usage:
            response = intelisys.chat("What is the capital of France?").get_response()
        """
        logger = logging.getLogger("chat")
        logger.info("Method called")
        logger.debug(f"User input: {user_input[:50]}...")
        if self.current_message:
            self.get_response()  # Send any pending message before starting a new one
        self.current_message = {"type": "text", "text": user_input}
        return self.get_response()

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
        self.logger.info(f"Image method called with path_or_url: {path_or_url}")
        if self.provider not in ["openai", "openrouter"]:
            raise ValueError("The image method is only supported for the OpenAI and OpenRouter providers.")
        
        if path_or_url.startswith(('http://', 'https://')):
            image_data = path_or_url
        else:
            # Validate local file path
            if not os.path.exists(path_or_url):
                raise FileNotFoundError(f"Image file not found: {path_or_url}")
            image_data = path_or_url

        self.image_urls.append(image_data)
        self.logger.debug(f"Added image: {path_or_url}")
        return self

    def get_response(self):
        """
        Get the AI's response for the current message.

        Returns:
            str or dict: The AI's response, which may be a string or a JSON object if json_mode is True.

        Raises:
            ValueError: If there's no message to send or if the response is None.

        Usage:
            response = intelisys.chat("Hello").get_response()
        """
        logger = logging.getLogger("get_response")
        logger.info("Method called")
        if self.current_message:
            self.add_message("user", self.current_message["text"])
        
        self.current_message = None
        
        if self.history:
            response = self._create_response(self.max_tokens or (4000 if self.provider != "anthropic" else 8192))
            self.last_response = self._handle_response(response)
        else:
            self.logger.warning("No message to send")
            self.last_response = None
        
        self.image_urls = []  # Clear image URLs after sending
        return self.last_response

    def _create_response(self, max_tokens, **kwargs):
        common_params = {
            "model": self.model,
            "messages": self.history.copy(),
            "stream": self.stream,
            "temperature": self.temperature,
        }

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

        self.logger.debug(f"API call params: {common_params}")

        if self.json_mode and self.provider == "openai":
            common_params["response_format"] = {"type": "json_object"}
        
        self.logger.debug(f"API call params: {common_params}")

        if self.provider == "anthropic":
            return self.client.messages.create(
                system=self.system_message,
                extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                **common_params,
                **kwargs
            )
        else:
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
        return response.content[0].text if self.provider == "anthropic" else response.choices[0].message.content

    def _extract_content(self, chunk):
        if self.provider == "anthropic":
            return chunk.delta.text if chunk.type == 'content_block_delta' else None
        return chunk.choices[0].delta.content if chunk.choices[0].delta.content else None

    def trim_history(self):
        self.logger.info("Trimming history")
        words_count = sum(len(str(m["content"]).split()) for m in self.history if m["role"] != "system")
        while words_count > self.max_history_words and len(self.history) > 1:
            removed_message = self.history.pop(0)
            words_count -= len(str(removed_message["content"]).split())
            self.logger.debug(f"Removed message from history: {removed_message['role']}")
        self.logger.debug(f"History trimmed. Current word count: {words_count}")
        return self

    def add_message(self, role, content):
        self.logger.info(f"Adding message with role: {role}")
        self.logger.debug(f"Message content: {content[:50]}...")  # Log first 50 chars
        if role == "user" and self.max_words_per_message:
            if isinstance(content, str):
                content += f" please use {self.max_words_per_message} words or less"
            elif isinstance(content, list) and content and isinstance(content[0], dict) and content[0].get('type') == 'text':
                content[0]['text'] += f" please use {self.max_words_per_message} words or less"

        self.history.append({"role": role, "content": content})
        return self

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
        """
        Send a chat message using a template and get the AI's response.

        Args:
            render_data (dict, optional): Data to render the template with.
            template (str, optional): The template string to use. If None, uses the default template.
            persona (str, optional): The persona to use for the system message. If None, uses the default persona.

        Returns:
            str or dict: The AI's response, which may be a string or a JSON object if json_mode is True.

        Raises:
            ValueError: If there's an error rendering the template.

        Usage:
            response = intelisys.template_chat(
                render_data={"name": "Alice", "question": "What's the weather like?"},
                template="Hello {{name}}, {{question}}",
                persona="You are a weather expert."
            )
        """
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

        self.last_response = response
        self.logger.info(f"template_chat response: {self.last_response}")

        return self.last_response

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

    # Async methods
    async def chat_async(self, user_input, **kwargs):
        self.logger.info("Async chat method called")
        await self.add_message_async("user", user_input)
        self.last_response = await self.get_response_async(**kwargs)
        return self.last_response

    async def add_message_async(self, role, content):
        self.logger.info(f"Async adding message with role: {role}")
        self.add_message(role, content)
        return self

    async def set_system_message_async(self, message=None):
        self.logger.info("Async setting system message")
        self.set_system_message(message)
        return self

    async def get_response_async(self, color=None, should_print=True, **kwargs):
        self.logger.info("Async get_response method called")
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
        self.logger.info("Async trimming history")
        self.trim_history()
        return self

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
        self.logger.info("Async template chat method called")
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
