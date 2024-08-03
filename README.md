# Intelisys

Intelisys is a powerful Python library that provides a unified interface for interacting with various AI models and services. It offers seamless integration with OpenAI, Anthropic, Google, TogetherAI, Groq, MistralAI, and more, making it an essential tool for AI-powered applications.

## New in Version 0.3.0

- Added support for multiple new AI models including OpenAI, Anthropic, Google, TogetherAI, Groq, and MistralAI
- Introduced asynchronous methods for chat and response handling
- Implemented template-based API calls with `template_api` and `template_api_json` functions
- Added a JSON fixing utility with the `fix_json` function
- Significantly refactored the `Intelisys` class for better performance and flexibility
- Improved error handling and logging across the library
- Enhanced API key management using 1Password Connect

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

- Multi-model support (OpenAI, Anthropic, Google, TogetherAI, Groq, MistralAI)
- Secure API key management with 1Password Connect
- Asynchronous and synchronous chat interfaces
- Template-based API calls for flexible prompts
- JSON formatting and fixing utilities
- Lazy loading of attributes for improved performance
- Comprehensive error handling and logging

## Quick Start

```python
from intelisys import Intelisys, get_completion_api

# Using Intelisys class
intelisys = Intelisys(name="MyAssistant", provider="openai", model="gpt-4")
response = intelisys.chat("Explain quantum computing")
print(response)

# Using get_completion_api
response = get_completion_api("What is machine learning?", "gpt-4")
print(response)
```

## Advanced Usage

```python
from intelisys import template_api_json, get_assistant, fix_json

# Template-based API call
render_data = {"topic": "artificial intelligence"}
system_message = "Explain {{topic}} in simple terms."
response = template_api_json("gpt-4", render_data, system_message, "teacher")
print(response)

# Using an OpenAI assistant
assistant_id = "your_assistant_id"
responses = get_assistant("Summarize the latest AI breakthroughs", assistant_id)
for response in responses:
    print(response)

# Fixing malformed JSON
malformed_json = "{'key': 'value', 'nested': {'a':1, 'b': 2,}}"
fixed_json = fix_json(malformed_json)
print(fixed_json)

# Asynchronous chat
import asyncio

async def async_chat():
    intelisys = Intelisys(name="AsyncAssistant", provider="anthropic", model="claude-3.5")
    response = await intelisys.chat_async("What are the implications of AGI?")
    print(response)

asyncio.run(async_chat())
```

## Supported Models

Intelisys supports a wide range of AI models:

- OpenAI: gpt-4o-mini, gpt-4, gpt-4o
- Anthropic: claude-3.5
- Google: gemini-flash
- TogetherAI: llama-3-70b, llama-3.1-large
- Groq: groq-llama, groq-fast
- MistralAI: mistral-large

## API Reference

For detailed information on available functions and their usage, please refer to the docstrings in the source code or our [API documentation](https://intelisys.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

For a detailed list of changes and version history, please refer to the [CHANGELOG.md](https://github.com/lifsys/intelisys/blob/main/CHANGELOG.md) file.

## About Lifsys, Inc

Lifsys, Inc is an innovative AI company dedicated to developing cutting-edge solutions for the future. Visit [www.lifsys.com](https://www.lifsys.com) to learn more about our mission and projects.
