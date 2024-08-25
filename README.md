# Intelisys

Intelisys is a powerful and flexible Python library that provides a unified interface for interacting with various AI providers and models. It simplifies the process of chatting with AI models, handling image inputs, and managing conversation history across different platforms.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install Intelisys, you can use pip:

```bash
pip install intelisys
```

## Usage

### Basic Usage

Here's a simple example of how to use Intelisys:

```python
from intelisys import Intelisys

# Initialize Intelisys
ai = Intelisys(provider="openai", model="gpt-3.5-turbo")

# Set a system message
ai.set_system_message("You are a helpful assistant.")

# Chat with the AI
response = ai.chat("Hello, how are you?")

# Get the AI's response
print(ai.get_response())
```

### Advanced Usage

Intelisys supports method chaining for a more fluent interface:

```python
from intelisys import Intelisys

ai = (Intelisys(provider="anthropic", model="claude-2")
      .set_system_message("You are a creative writing assistant.")
      .set_default_template("User: {}\nAI: ")
      .set_default_persona("You are a witty and sarcastic AI."))

response = ai.chat("Write a short story about a time-traveling toaster.")
print(ai.get_response())
```

### Asynchronous Usage

Intelisys also supports asynchronous operations:

```python
import asyncio
from intelisys import Intelisys

async def main():
    ai = Intelisys(provider="openai", model="gpt-4")
    await ai.set_system_message_async("You are a helpful assistant.")
    await ai.chat_async("What's the capital of France?")
    response = await ai.get_response_async()
    print(response)

asyncio.run(main())
```

## API Reference

### Class: Intelisys

#### `__init__(self, provider: str, model: str = None, **kwargs)`

Initializes an Intelisys instance.

Parameters:
- `provider` (str): The AI provider to use (e.g., "openai", "anthropic")
- `model` (str, optional): The specific model to use. If not provided, a default model for the provider will be used.
- `**kwargs`: Additional keyword arguments for configuration.

#### `set_log_level(self, level: Union[int, str])`

Sets the log level for the Intelisys instance.

Parameters:
- `level` (Union[int, str]): The log level to set (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

#### `set_system_message(self, message=None)`

Sets the system message for the conversation.

Parameters:
- `message` (str, optional): The system message to set. If None, clears the current system message.

#### `chat(self, user_input)`

Sends a user message to the AI and gets a response.

Parameters:
- `user_input` (str): The user's input message.

Returns:
- The AI's response.

#### `get_response(self)`

Retrieves the latest response from the AI.

Returns:
- The AI's latest response.

#### `trim_history(self)`

Trims the conversation history to stay within token limits.

#### `add_message(self, role, content)`

Adds a message to the conversation history.

Parameters:
- `role` (str): The role of the message sender (e.g., "user", "assistant", "system")
- `content` (str): The content of the message.

#### `set_default_template(self, template: str) -> 'Intelisys'`

Sets a default template for formatting messages.

Parameters:
- `template` (str): The template string to use.

Returns:
- The Intelisys instance for method chaining.

#### `set_default_persona(self, persona: str) -> 'Intelisys'`

Sets a default persona for the AI.

Parameters:
- `persona` (str): The persona description.

Returns:
- The Intelisys instance for method chaining.

### Asynchronous Methods

Intelisys provides asynchronous versions of most methods, suffixed with `_async`. These include:

- `chat_async(self, user_input, **kwargs)`
- `add_message_async(self, role, content)`
- `set_system_message_async(self, message=None)`
- `get_response_async(self, color=None, should_print=True, **kwargs)`
- `trim_history_async(self)`

These methods work similarly to their synchronous counterparts but should be used with `async/await` syntax.

## Contributing

We welcome contributions to Intelisys! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and write tests if applicable
4. Run the existing tests to ensure nothing was broken
5. Commit your changes and push to your fork
6. Create a pull request with a clear description of your changes

Please ensure your code adheres to the existing style and passes all tests before submitting a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
