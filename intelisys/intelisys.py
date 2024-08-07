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
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

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

def iterative_llm_fix_json(json_str: str, max_attempts: int = 5) -> str:
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
            fixed_json = Intelisys(
                provider="openai", 
                model="gpt-4o-mini",
                json_mode=True) \
                .set_system_message("Correct the JSON and return only the fixed JSON.") \
                .chat(f"{prompt}\n\n{json_str}") \
                .results()
            json.loads(fixed_json)  # Validate the JSON
            return fixed_json
        except json.JSONDecodeError as e:
            line_no, col_no, context = locate_json_error(fixed_json, str(e))
            print(f"Fix attempt failed. Error at line {line_no}, column {col_no}:\n{context}")

    raise ValueError("Failed to fix JSON after multiple attempts")

def safe_json_loads(json_str: str, error_prefix: str = "") -> Dict:
    """Safely load JSON string, with iterative LLM-based error correction."""
    json_str = remove_preface(json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        line_no, col_no, context = locate_json_error(json_str, str(e))
        print(f"{error_prefix}Initial JSON parsing failed at line {line_no}, column {col_no}:\n{context}")
        
        fix_attempts = [
            iterative_llm_fix_json,
            lambda s: Intelisys(
                provider="openai", 
                model="gpt-4o-mini",
                json_mode=True) \
                .set_system_message("Return only the fixed JSON.") \
                .chat(f"Fix this JSON:\n{s}") \
                .results(),
            ast.literal_eval
        ]
        
        for fix in fix_attempts:
            try:
                fixed_json = fix(json_str)
                return json.loads(fixed_json) if isinstance(fixed_json, str) else fixed_json
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
        
        print(f"{error_prefix}JSON parsing failed after all correction attempts.")
        print(f"Problematic JSON string: {json_str}")
        raise ValueError(f"{error_prefix}Failed to parse JSON after multiple attempts.")

class Intelisys:
    SUPPORTED_PROVIDERS = {"openai", "anthropic", "openrouter", "groq"}
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20240620",
        "openrouter": "meta-llama/llama-3.1-405b-instruct",
        "groq": "llama-3.1-8b-instant"
    }

    def __init__(self, name="Intelisys", api_key=None, max_history_words=10000,
                 max_words_per_message=None, json_mode=False, stream=False, use_async=False,
                 max_retry=10, provider="anthropic", model=None, should_print_init=False,
                 print_color="green", temperature=0, max_tokens=None, log: Union[str, int] = "WARNING"):
        
        # Set up logger
        self.logger = logging.getLogger(f"{name}")
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
        
        if should_print_init:
            print(colored(f"\n{self.name} initialized with provider={self.provider}, model={self.model}, json_mode={self.json_mode}, temp={self.temperature}", "red"))

        self.logger.debug(f"Intelisys initialized with: name={name}, max_history_words={max_history_words}, "
                          f"max_words_per_message={max_words_per_message}, json_mode={json_mode}, "
                          f"stream={stream}, use_async={use_async}, max_retry={max_retry}, "
                          f"temperature={temperature}, max_tokens={max_tokens}")

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
        self.system_message = message or "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode and "json" not in message.lower():
            self.system_message += " Please return your response in JSON unless user has specified a system message."
        self.logger.info(f"System message set: {self.system_message[:50]}...")  # Log first 50 chars
        return self

    def chat(self, user_input):
        self.logger.info(f"Chat method called")
        self.logger.debug(f"User input: {user_input[:50]}...")  # Log first 50 chars
        if self.current_message or self.image_urls:
            self.send()  # Send any pending message before starting a new one
        self.current_message = {"type": "text", "text": user_input}
        return self

    def _encode_image(self, image_path: str) -> str:
        self.logger.debug(f"Encoding image: {image_path}")
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='PNG')
            return base64.b64encode(byte_arr.getvalue()).decode('utf-8')

    def image(self, path_or_url: str, detail: str = "auto"):
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
        
        self.image_urls.append(image_data)
        self.logger.debug(f"Added image: {path_or_url}")
        return self

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

    def get_response(self):
        self.logger.info("get_response method called")
        max_tokens = self.max_tokens if self.max_tokens is not None else (4000 if self.provider != "anthropic" else 8192)

        for attempt in range(self.max_retry):
            try:
                self.logger.debug(f"Attempt {attempt + 1} to get response")
                response = self._create_response(max_tokens)
                
                assistant_response = self._handle_stream(response, self.print_color, True) if self.stream else self._handle_non_stream(response)

                if self.json_mode:
                    if self.provider == "openai":
                        try:
                            assistant_response = json.loads(assistant_response)
                        except json.JSONDecodeError as json_error:
                            self.logger.error(f"JSON decoding error: {json_error}")
                            raise
                    else:
                        assistant_response = safe_json_loads(assistant_response, error_prefix="Intelisys JSON parsing: ")

                self.add_message("assistant", str(assistant_response))
                self.trim_history()
                return assistant_response
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1}/{self.max_retry}: {e}")
                if attempt < self.max_retry - 1:
                    import time
                    time.sleep(1)
                else:
                    raise Exception(f"Max retries reached. Last error: {e}")

    def _create_response(self, max_tokens, **kwargs):
        self.logger.debug(f"Creating response with max_tokens={max_tokens}")
        if self.provider == "anthropic":
            return self.client.messages.create(
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
            
            self.logger.debug(f"API call params: {common_params}")
            return self.client.chat.completions.create(**common_params)

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
                    persona: Optional[str] = None, 
                    return_self: bool = True) -> Union['Intelisys', Dict]:
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
        response = self.chat(prompt).send().results()  # Call results() here

        self.last_response = response
        self.logger.info(f"template_chat response: {self.last_response}")

        if return_self:
            return self
        else:
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
        return self

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
        response = self.results()

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
