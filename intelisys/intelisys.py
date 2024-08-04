"""
Provides intelligence/AI services for the Lifsys Enterprise.

This module requires a 1Password Connect server to be available and configured.
The OP_CONNECT_TOKEN and OP_CONNECT_HOST environment variables must be set
for the onepasswordconnectsdk to function properly.
"""
import asyncio
import json
import os
from typing import Dict,Optional, Union
from contextlib import contextmanager

from anthropic import Anthropic, AsyncAnthropic
from jinja2 import Template
from openai import AsyncOpenAI, OpenAI
from termcolor import colored
from utilisys import safe_json_loads

class Intelisys:
    SUPPORTED_PROVIDERS = {"openai", "anthropic", "openrouter", "groq"}
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20240620",
        "openrouter": "meta-llama/llama-3.1-405b-instruct",
        "groq": "llama-3.1-8b-instant"
    }

    def __init__(self, name="Intelisys", api_key=None, max_history_words=10000,
                 max_words_per_message=None, json_mode=False, stream=True, use_async=False,
                 max_retry=10, provider="anthropic", model=None, should_print_init=False,
                 print_color="green", temperature=0):
        
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

    def add_message(self, role, content):
        if role == "user" and self.max_words_per_message:
            content += f" please use {self.max_words_per_message} words or less"
        self.history.append({"role": role, "content": str(content)})
        return self

    def chat(self, user_input, **kwargs):
        self.add_message("user", user_input)
        kwargs.pop('provider', None)
        self.last_response = self.get_response(**kwargs)
        return self

    def get_response(self, color=None, should_print=True, **kwargs):
        color = color or self.print_color
        max_tokens = kwargs.pop('max_tokens', 4000 if self.provider != "anthropic" else 8192)

        for attempt in range(self.max_retry):
            try:
                response = self._create_response(max_tokens, **kwargs)
                
                assistant_response = self._handle_stream(response, color, should_print) if self.stream else self._handle_non_stream(response)

                if self.json_mode:
                    if self.provider == "openai":
                        try:
                            assistant_response = json.loads(assistant_response)
                        except json.JSONDecodeError as json_error:
                            print(f"JSON decoding error: {json_error}")
                            raise
                    else:
                        assistant_response = safe_json_loads(assistant_response, error_prefix="Intelisys JSON parsing: ")

                self.add_message("assistant", str(assistant_response))
                self.trim_history()
                return assistant_response
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{self.max_retry}: {e}")
                if attempt < self.max_retry - 1:
                    import time
                    time.sleep(1)
                else:
                    raise Exception(f"Max retries reached. Last error: {e}")

    def _create_response(self, max_tokens, **kwargs):
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
        """
        Returns the last response from the AI model.

        Returns:
            The last response from the AI model, which could be a string or a dictionary (if in JSON mode).
        """
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
