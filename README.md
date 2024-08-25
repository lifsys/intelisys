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

#### `set_log_level(self, level: Union[int, str])`
Sets the log level for the Intelisys instance.

#### `set_system_message(self, message=None)`
Sets the system message for the conversation.

#### `chat(self, user_input)`
Sends a user message to the AI and gets a response.

#### `get_response(self)`
Retrieves the latest response from the AI.

#### `trim_history(self)`
Trims the conversation history to stay within token limits.

#### `add_message(self, role, content)`
Adds a message to the conversation history.

#### `set_default_template(self, template: str) -> 'Intelisys'`
Sets a default template for formatting messages.

#### `set_default_persona(self, persona: str) -> 'Intelisys'`
Sets a default persona for the AI.

#### `remove_preface(text: str) -> str`
Removes preface from the given text.

#### `safe_json_loads(json_str: str, error_prefix: str = "") -> Dict`
Safely loads a JSON string into a dictionary.

#### `chat_async(self, user_input, **kwargs)`
Asynchronous version of chat method.

#### `add_message_async(self, role, content)`
Asynchronous version of add_message method.

#### `set_system_message_async(self, message=None)`
Asynchronous version of set_system_message method.

#### `get_response_async(self, color=None, should_print=True, **kwargs)`
Asynchronous version of get_response method.

#### `trim_history_async(self)`
Asynchronous version of trim_history method.

For detailed information on parameters and return values, please refer to the method docstrings in the source code.

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
