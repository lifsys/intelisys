# Intelisys

Intelisys is a powerful Python library that provides a unified interface for interacting with various AI models and services. It offers seamless integration with OpenAI, Anthropic, OpenRouter, and Groq, making it an essential tool for AI-powered applications.

## New in Version 0.4.0

- Major version update to 0.4.0
- Significant improvements and new features added to the main code
- Enhanced error handling and logging capabilities
- Improved JSON parsing with fallback to safe_json_loads
- Added support for image input in chat method
- Implemented clear() method for resetting current message and image URLs
- Updated default models for various providers
- Improved async methods for better performance
- Expanded documentation to cover new features and improvements
- Ensured consistency in version numbers across all package files

## Installation

Install Intelisys using pip:

```
pip install intelisys
```

For the latest development version:

```
pip install git+https://github.com/lifsys/intelisys.git
```

## Requirements

- Python 3.7 or higher
- A 1Password Connect server (for API key management)
- Environment variables:
  - `OP_CONNECT_TOKEN`: Your 1Password Connect token
  - `OP_CONNECT_HOST`: The URL of your 1Password Connect server

**Note**: The library requires a local 1Password Connect server for API key retrieval.

## Key Features

- Multi-provider support (OpenAI, Anthropic, OpenRouter, Groq)
- Secure API key management with 1Password Connect
- Asynchronous and synchronous chat interfaces
- Template-based API calls for flexible prompts
- JSON mode support for structured responses
- Lazy loading of attributes for improved performance
- Comprehensive error handling and logging
- Retry mechanism for API calls

## Quick Start

```python
from intelisys import Intelisys

# Basic usage
intelisys = Intelisys(name="MyAssistant", provider="openai", model="gpt-4")
response = intelisys.chat("Explain quantum computing")
print(response)

# Chaining example
result = (Intelisys(provider="openai", model="gpt-4")
    .chat("Explain the concept of")
    .chat("quantum entanglement")
    .chat("in simple terms"))
print(result)

# JSON mode
intelisys_json = Intelisys(name="JSONAssistant", provider="openai", model="gpt-4", json_mode=True)
response = intelisys_json.chat("List 3 quantum computing concepts")
print(response)  # This will be a Python dictionary

# Image OCR example
intelisys = Intelisys(provider="openai", model="gpt-4-vision-preview")
result = (intelisys
    .chat("Please provide all the text in the following image(s).")
    .image("http://www.mattmahoney.net/ocr/stock_gs200.jpg")
    .image("/path/to/local/image.png"))
print(result)
```

## Advanced Usage

```python
from intelisys import Intelisys
import asyncio

# Template-based API call
intelisys = Intelisys(name="TemplateAssistant", provider="anthropic", model="claude-3-opus-20240229")
render_data = {"topic": "artificial intelligence"}
template = "Explain {{topic}} in simple terms."
response = intelisys.template_chat(render_data, template)
print(response)

# Asynchronous chat
async def async_chat():
    intelisys = Intelisys(name="AsyncAssistant", provider="anthropic", model="claude-3-opus-20240229")
    response = await intelisys.chat_async("What are the implications of AGI?")
    print(response)

asyncio.run(async_chat())

# Using context manager for temporary template and persona changes
intelisys = Intelisys(name="ContextAssistant", provider="openai", model="gpt-4")
with intelisys.template_context(template="Summarize {{topic}} in one sentence.", persona="You are a concise summarizer."):
    response = intelisys.template_chat({"topic": "quantum entanglement"})
    print(response)

# Using retry mechanism
intelisys = Intelisys(name="RetryAssistant", provider="openai", model="gpt-4", max_retry=5)
response = intelisys.chat("This might fail, but we'll retry")
print(response)
```

## Callable Elements

The Intelisys class provides the following main callable elements:

1. `chat(user_input)`: Adds a user message to the conversation.
2. `image(path_or_url)`: Adds an image to the conversation (for vision models).
3. `send()`: Sends the current conversation to the AI model.
4. `results()`: Retrieves the results of the last API call.
5. `set_system_message(message)`: Sets the system message for the conversation.
6. `template_chat(render_data, template, persona)`: Performs a template-based chat.
7. `set_default_template(template)`: Sets the default template for template-based chats.
8. `set_default_persona(persona)`: Sets the default persona for template-based chats.
9. `clear()`: Clears the current message and image URLs without sending.
10. `trim_history()`: Trims the conversation history to stay within token limits.

Async versions of some methods are also available, such as `chat_async()`, `template_chat_async()`, etc.

For a complete list of methods and their descriptions, please refer to the source code or the API documentation.

## Supported Providers and Models

Intelisys supports a wide range of AI providers and models:

- OpenAI: Various GPT models including gpt-4
- Anthropic: Claude models including claude-3-5-sonnet-20240620
- OpenRouter: Access to multiple AI models through a single API
- Groq: Fast inference models

For a complete list of supported models, please refer to the `DEFAULT_MODELS` dictionary in the `Intelisys` class.

## Error Handling

Intelisys now includes improved error handling and a retry mechanism for API calls. If an API call fails, the library will automatically retry the call up to the specified `max_retry` times (default is 10). This helps to handle temporary network issues or API rate limits.

## JSON Parsing

For JSON responses, Intelisys now uses a more robust parsing method. If the standard `json.loads()` fails, it falls back to `safe_json_loads()` from the `utilisys` library, which can handle some common JSON parsing errors.

## API Reference

For detailed information on available methods and their usage, please refer to the docstrings in the source code or our [API documentation](https://intelisys.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

For a detailed list of changes and version history, please refer to the [CHANGELOG.md](https://github.com/lifsys/intelisys/blob/main/CHANGELOG.md) file.

## About Lifsys, Inc

Lifsys, Inc is an innovative AI company dedicated to developing cutting-edge solutions for the future. Visit [www.lifsys.com](https://www.lifsys.com) to learn more about our mission and projects.
