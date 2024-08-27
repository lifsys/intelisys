# Intelisys: Your Advanced AI Assistant Library

Welcome to Intelisys, a powerful and versatile AI assistant library that provides a unified interface for interacting with various AI models and providers. Intelisys is designed to make AI integration seamless and efficient for developers of all levels.

## What's New in Version 0.5.6

- Enhanced `max_history_words` functionality for improved conversation management
- Optimized history trimming for efficient memory usage
- Improved asynchronous support with better error handling
- Added support for structured output with OpenAI provider using Pydantic models
- Expanded reference capabilities, now supporting various document types (PDF, Word, Excel, PowerPoint)
- Improved performance for long-running conversations

## Key Features

- Multi-provider support (OpenAI, Anthropic, OpenRouter, Groq)
- Asynchronous operations for improved performance
- Template-based chat for easy customization
- Structured output support using Pydantic models (OpenAI only)
- Image processing capabilities
- Efficient conversation history management
- Reference information support from various sources

## Installation

Install Intelisys using pip:

```bash
pip install intelisys
```

## Quick Start

Here's a simple example to get you started:

```python
from intelisys import Intelisys

# Create an Intelisys instance
ai = Intelisys(provider="openai", model="gpt-4")

# Chat with the AI
response = ai.chat("What is the capital of France?")
print(response)
```

## Advanced Usage

### Template-Based Chat

Use templates for more structured interactions:

```python
ai = Intelisys(provider="anthropic", model="claude-3-5-sonnet-20240620")
ai.set_default_template("Explain {{topic}} in simple terms.")
response = ai.template_chat(render_data={"topic": "quantum computing"})
print(response)
```

### Asynchronous Operations

Perform asynchronous chats for improved efficiency:

```python
import asyncio

async def async_chat():
    ai = Intelisys(provider="openai", model="gpt-4", use_async=True)
    response = await ai.chat_async("Discuss the future of AI")
    print(response)

asyncio.run(async_chat())
```

### Structured Output

Get structured responses using Pydantic models (OpenAI only):

```python
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

ai = Intelisys(provider="openai", model="gpt-4")
ai.set_output_model(MovieReview)
result = ai.chat("Review the movie 'Inception'")
print(result)  # This will be a MovieReview instance
```

### Reference Information

Provide context to your AI assistant:

```python
ai = Intelisys(provider="openai", model="gpt-4")
ai.reference("https://example.com/article.html")
ai.reference("/path/to/local/document.pdf")
response = ai.chat("Summarize the referenced information")
print(response)
```

## API Reference

For a complete API reference, please refer to our [documentation](https://intelisys.readthedocs.io/).

## Contributing

We welcome contributions to Intelisys! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

For a detailed list of changes and version history, please refer to the [CHANGELOG.md](CHANGELOG.md) file.
