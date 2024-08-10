# Intelisys

Intelisys is a powerful Python library that provides a unified interface for interacting with various AI models and services. It offers seamless integration with OpenAI, Anthropic, OpenRouter, and Groq, making it an essential tool for AI-powered applications.

## Changelog

### [0.5.0] - 2024-08-21
- Incremented version number to 0.5.0
- Significant improvements to max_history_words functionality and trimming
- Enhanced logging for better debugging and monitoring
- Updated PyPI package configuration for smoother installation
- Refined error handling for more robust operation
- Updated documentation to reflect new changes and features
- Ensured version consistency across all package files

### [0.4.9] - 2024-08-20
- Improved logging functionality
- Updated PyPI package configuration
- Enhanced error handling
- Refined documentation in README.md
- Ensured version consistency across all files
- Updated installation and usage instructions

[All previous changelog entries remain unchanged]

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

# Using Intelisys class
intelisys = Intelisys(name="MyAssistant", provider="openai", model="gpt-4")
response = intelisys.chat("Explain quantum computing").get_response()
print(response)

# Using JSON mode
intelisys_json = Intelisys(name="JSONAssistant", provider="openai", model="gpt-4", json_mode=True)
response = intelisys_json.chat("List 3 quantum computing concepts").get_response()
print(response)  # This will be a Python dictionary

# Image OCR example
intelisys = Intelisys(provider="openai", model="gpt-4-vision-preview")
result = (intelisys
 .chat("Please provide all the text in the following image(s).")
 .image("http://www.mattmahoney.net/ocr/stock_gs200.jpg")
 .image("/Users/lifsys/Documents/devhub/testingZone/_Archive/screen_small-2.png")
 .get_response()
)
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
response = intelisys.template_chat(render_data, template).get_response()
print(response)

# Asynchronous chat
async def async_chat():
    intelisys = Intelisys(name="AsyncAssistant", provider="anthropic", model="claude-3-opus-20240229")
    response = await intelisys.chat_async("What are the implications of AGI?")
    print(await response.get_response_async())

asyncio.run(async_chat())

# Using context manager for temporary template and persona changes
intelisys = Intelisys(name="ContextAssistant", provider="openai", model="gpt-4")
with intelisys.template_context(template="Summarize {{topic}} in one sentence.", persona="You are a concise summarizer."):
    response = intelisys.template_chat({"topic": "quantum entanglement"}).get_response()
    print(response)

# Using retry mechanism
intelisys = Intelisys(name="RetryAssistant", provider="openai", model="gpt-4", max_retry=5)
response = intelisys.chat("This might fail, but we'll retry").get_response()
print(response)
```

## Supported Providers and Models

Intelisys supports a wide range of AI providers and models:

- OpenAI: Various GPT models including gpt-4 and gpt-4-vision-preview
- Anthropic: Claude models including claude-3-opus-20240229
- OpenRouter: Access to multiple AI models through a single API
- Groq: Fast inference models

For a complete list of supported models, please refer to the `DEFAULT_MODELS` dictionary in the `Intelisys` class.

## Error Handling

Intelisys includes improved error handling and a retry mechanism for API calls. If an API call fails, the library will automatically retry the call up to the specified `max_retry` times (default is 10). This helps to handle temporary network issues or API rate limits.

## JSON Parsing

For JSON responses, Intelisys uses a robust parsing method. If the standard `json.loads()` fails, it falls back to `safe_json_loads()`, which can handle some common JSON parsing errors.

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
