# Intelisys

Intelisys is a Python library that provides intelligence/AI services for the Lifsys Enterprise. It offers a unified interface to interact with various AI models and services, including OpenAI, Anthropic, Google, and more.

## Installation

You can install Intelisys using pip:

```
pip install intelisys
```

This will install the latest stable version from PyPI.

For the latest development version, you can install directly from GitHub:

```
pip install git+https://github.com/lifsys/intelisys.git
```

Note: If you encounter any issues during installation, make sure you have the latest version of pip:

```
pip install --upgrade pip
```

Then try the installation again.

## Requirements

- Python 3.7 or higher
- A 1Password Connect server (for API key management)
- Environment variables:
  - `OP_CONNECT_TOKEN`: Your 1Password Connect token
  - `OP_CONNECT_HOST`: The URL of your 1Password Connect server

**Note**: If no local 1Password Connect server is available, the library will fail to retrieve API keys.

## Features

- Support for multiple AI models (OpenAI, Anthropic, Google, TogetherAI, Groq, MistralAI)
- Secure API key management using 1Password Connect
- JSON formatting and template rendering
- Asynchronous assistant interactions
- Template-based API calls
- Improved error handling and logging

## Usage

Here's a quick example of how to use Intelisys:

```python
from intelisys import get_completion_api

# Make sure OP_CONNECT_TOKEN and OP_CONNECT_HOST are set in your environment

response = get_completion_api("Hello, how are you?", "gpt-4")
print(response)
```

### Advanced Usage

```python
from intelisys import template_api_json, get_assistant, fix_json

# Using a template for API calls
render_data = {"user_name": "Alice"}
system_message = "You are a helpful assistant. Greet {{user_name}}."
response = template_api_json("gpt-4", render_data, system_message, "friendly_assistant")
print(response)

# Using an OpenAI assistant
assistant_id = "your_assistant_id"
reference = "What's the weather like today?"
responses = get_assistant(reference, assistant_id)
for response in responses:
    print(response)

# Fixing malformed JSON
malformed_json = "{'key': 'value', 'nested': {'a':1, 'b': 2,}}"
fixed_json = fix_json(malformed_json)
print(fixed_json)
```

## Supported Models

Intelisys supports a variety of AI models:

- OpenAI: gpt-4o-mini, gpt-4, gpt-4o
- Anthropic: claude-3.5
- Google: gemini-flash
- TogetherAI: llama-3-70b, llama-3.1-large
- Groq: groq-llama, groq-fast
- MistralAI: mistral-large

## New in Version 0.2.1

- Fixed minor bugs and improved stability
- Updated documentation to reflect the latest changes
- Enhanced compatibility with the latest versions of dependent libraries

## API Reference

For detailed information on available functions and their usage, please refer to the docstrings in the source code.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

For a detailed list of changes and version history, please refer to the [CHANGELOG.md](https://github.com/lifsys/intelisys/blob/main/CHANGELOG.md) file.

## About Lifsys, Inc

Lifsys, Inc is an AI company dedicated to developing solutions for the future. For more information, visit [www.lifsys.com](https://www.lifsys.com).
