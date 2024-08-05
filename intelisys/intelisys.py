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
from typing import Dict,Optional, Union, Tuple
from contextlib import contextmanager
from PIL import Image
from anthropic import Anthropic, AsyncAnthropic
from jinja2 import Template
from openai import AsyncOpenAI, OpenAI
from termcolor import colored
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def remove_preface(text: str) -> str:
    """
    Remove any prefaced text before the start of JSON content.

    Args:
    text (str): The input text that may contain prefaced content before JSON.

    Returns:
    str: The text with any preface removed, starting from the first valid JSON character.
    """
    match: Optional[re.Match] = re.search(r"[\{\[]", text)
    
    if match:
        start: int = match.start()
        if start > 0:
            logger.info(f"Removed preface of length {start} characters")
        return text[start:]
    
    logger.warning("No JSON-like content found in the text")
    return text

def locate_json_error(json_str: str, error_msg: str) -> Tuple[int, int, str]:
    """
    Locate the position of the JSON error and return the surrounding context.

    Args:
    json_str (str): The JSON string with the error.
    error_msg (str): The error message from json.JSONDecodeError.

    Returns:
    Tuple[int, int, str]: Line number, column number, and the problematic part of the JSON string.
    """
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
            logger.warning(f"Fix attempt failed. Error at line {line_no}, column {col_no}:\n{context}")

    raise ValueError("Failed to fix JSON after multiple attempts")

def safe_json_loads(json_str: str, error_prefix: str = "") -> Dict:
    """Safely load JSON string, with iterative LLM-based error correction."""
    json_str = remove_preface(json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        line_no, col_no, context = locate_json_error(json_str, str(e))
        logger.warning(f"{error_prefix}Initial JSON parsing failed at line {line_no}, column {col_no}:\n{context}")
        
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
        
        logger.error(f"{error_prefix}JSON parsing failed after all correction attempts.")
        logger.debug(f"Problematic JSON string: {json_str}")
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
                 print_color="green", temperature=0, max_tokens=None):
        
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
        
        self.logger = logging.getLogger(__name__)
      
        if should_print_init:
            print(colored(f"\n{self.name} initialized with provider={self.provider}, model={self.model}, json_mode={self.json_mode}, temp={self.temperature}", "red"))

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

    def set_system_message(self, message=None):
        self.system_message = message or "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode and "json" not in message.lower():
            self.system_message += " Please return your response in JSON unless user has specified a system message."
        return self

    def chat(self, user_input):
        self.logger.debug(f"Chat method called with user input: {user_input}")
        if self.current_message or self.image_urls:
            self.send()  # Send any pending message before starting a new one
        self.current_message = {"type": "text", "text": user_input}
        return self

    def _encode_image(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            # Convert to RGB mode if it's not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create a byte stream
            byte_arr = io.BytesIO()
            # Save as PNG for lossless compression
            img.save(byte_arr, format='PNG')
            # Get the byte string and encode to base64
            return base64.b64encode(byte_arr.getvalue()).decode('utf-8')

    def image(self, path_or_url: str, detail: str = "auto"):
        if self.provider not in ["openai", "openrouter"]:
            raise ValueError("The image method is only supported for the OpenAI and OpenRouter providers.")
        
        if os.path.isfile(path_or_url):
            # It's a local file path
            encoded_image = self._encode_image(path_or_url)
            image_data = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}",
                    "detail": detail
                }
            }
        else:
            # It's a URL
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
        self.logger.debug(f"Send method called")
        self.logger.debug(f"Current message: {self.current_message}")
        self.logger.debug(f"Current image URLs: {self.image_urls}")
        
        if self.provider in ["openai", "openrouter"]:
            content = []
            if self.current_message:
                content.append(self.current_message)
            content.extend(self.image_urls)
            if content:
                self.add_message("user", content)
                self.logger.debug(f"Added message with content: {content}")
        else:
            if self.current_message:
                self.add_message("user", self.current_message["text"])
        
        self.current_message = None
        self.image_urls = []  # Clear the image URLs after adding them to the history
        self.logger.debug("Cleared current message and image URLs")
        
        if self.history:
            self.last_response = self.get_response()
        else:
            self.logger.warning("No message to send")
            self.last_response = None
        
        return self

    def clear(self):
        self.current_message = None
        self.image_urls = []
        self.logger.debug("Cleared current message and image URLs without sending")
        return self
    def add_message(self, role, content):
        if role == "user" and self.max_words_per_message:
            if isinstance(content, str):
                content += f" please use {self.max_words_per_message} words or less"
            elif isinstance(content, list) and content and isinstance(content[0], dict) and content[0].get('type') == 'text':
                content[0]['text'] += f" please use {self.max_words_per_message} words or less"

        self.history.append({"role": role, "content": content})
        self.logger.debug(f"Added message: role={role}, content={content}")
        return self

    def get_response(self):
        self.logger.debug("get_response method called")
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
        return response.content[0].text if self.provider == "anthropic" else response.choices[0].message.content

    def _extract_content(self, chunk):
        if self.provider == "anthropic":
            return chunk.delta.text if chunk.type == 'content_block_delta' else None
        return chunk.choices[0].delta.content if chunk.choices[0].delta.content else None

    def trim_history(self):
        words_count = sum(len(str(m["content"]).split()) for m in self.history if m["role"] != "system")
        while words_count > self.max_history_words and len(self.history) > 1:
            words_count -= len(self.history.pop(0)["content"].split())
        return self

    def results(self):
        if self.last_response is None:
            self.last_response = self.send()
        return self.last_response


    def set_default_template(self, template: str) -> 'Intelisys':
        self.default_template = template
        return self

    def set_default_persona(self, persona: str) -> 'Intelisys':
        self.default_persona = persona
        return self

    def set_template_instruction(self, set: str, instruction: str):
        self.template_instruction = self._go_get_api(set, instruction, "Promptsys")
        return self

    def set_template_persona(self, persona: str):
        self.template_persona = self._go_get_api("persona", persona, "Promptsys")
        return self

    def set_template_data(self, render_data: Dict):
        self.template_data = render_data
        return self

    def template_chat(self, 
                    render_data: Optional[Dict[str, Union[str, int, float]]] = None, 
                    template: Optional[str] = None, 
                    persona: Optional[str] = None, 
                    parse_json: bool = False) -> 'Intelisys':
        try:
            template = Template(template or self.default_template)
            merged_data = {**self.template_data, **(render_data or {})}
            prompt = template.render(**merged_data)
        except Exception as e:
            raise ValueError(f"Invalid template: {e}")

        self.set_system_message(persona or self.default_persona)
        response = self.chat(prompt).results()

        if self.json_mode:
            if isinstance(response, dict):
                self.last_response = response
            elif isinstance(response, str):
                try:
                    self.last_response = json.loads(response)
                except json.JSONDecodeError:
                    self.last_response = safe_json_loads(response, error_prefix="Intelisys template chat JSON parsing: ")
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
        else:
            self.last_response = response
        
        return self

    @contextmanager
    def template_context(self, template: Optional[str] = None, persona: Optional[str] = None):
        old_template, old_persona = self.default_template, self.default_persona
        if template:
            self.set_default_template(template)
        if persona:
            self.set_default_persona(persona)
        try:
            yield
        finally:
            self.default_template, self.default_persona = old_template, old_persona

    # Async methods
    async def chat_async(self, user_input, **kwargs):
        await self.add_message_async("user", user_input)
        self.last_response = await self.get_response_async(**kwargs)
        return self

    async def add_message_async(self, role, content):
        self.add_message(role, content)
        return self

    async def set_system_message_async(self, message=None):
        self.set_system_message(message)
        return self

    async def get_response_async(self, color=None, should_print=True, **kwargs):
        color = color or self.print_color
        max_tokens = kwargs.pop('max_tokens', 4000 if self.provider != "anthropic" else 8192)

        response = await self._create_response_async(max_tokens, **kwargs)

        assistant_response = await self._handle_stream_async(response, color, should_print) if self.stream else await self._handle_non_stream_async(response)

        if self.json_mode and self.provider == "openai":
            try:
                assistant_response = json.loads(assistant_response)
            except json.JSONDecodeError as json_error:
                print(f"JSON decoding error: {json_error}")
                raise

        await self.add_message_async("assistant", str(assistant_response))
        await self.trim_history_async()
        return assistant_response

    async def _create_response_async(self, max_tokens, **kwargs):
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
        return response.content[0].text if self.provider == "anthropic" else response.choices[0].message.content

    def _extract_content_async(self, chunk):
        if self.provider == "anthropic":
            return chunk.delta.text if chunk.type == 'content_block_delta' else None
        return chunk.choices[0].delta.content if chunk.choices[0].delta.content else None

    async def trim_history_async(self):
        self.trim_history()
        return self

    async def template_chat_async(self, 
                                render_data: Optional[Dict[str, Union[str, int, float]]] = None, 
                                template: Optional[str] = None, 
                                persona: Optional[str] = None, 
                                parse_json: bool = False) -> 'Intelisys':
        try:
            template = Template(template or self.default_template)
            merged_data = {**self.template_data, **(render_data or {})}
            prompt = template.render(**merged_data)
        except Exception as e:
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
                raise ValueError(f"Unexpected response type: {type(response)}")
        else:
            self.last_response = response
        
        return self
