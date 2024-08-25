# Intelisys: Your Friendly AI Assistant Library

Hello there, future AI master! 👋 Let's embark on an exciting journey to learn about Intelisys, a magical tool that helps you talk to different AI friends. Imagine having a universal remote control for all your AI toys – that's what Intelisys is!

## What is Intelisys?

Intelisys is like a super-smart translator that helps you chat with different AI friends (we call them "providers") using just one language. It's like having a friend who speaks many languages and can help you talk to people from all over the world!

## Getting Started

### Step 1: Install Intelisys

First, we need to invite Intelisys to our playground. Open your computer's command center (we call it a "terminal") and type:

```bash
pip install intelisys
```

### Step 2: Import Intelisys

Now that we have Intelisys installed, let's bring it into our code playground. In your Python code, add the following line:

```python
from intelisys import Intelisys
```

### Step 3: Create an Intelisys Instance

Next, we need to create an instance of Intelisys. Think of this like creating a new friend object. We'll give it a name, and tell it which AI friend we want to talk to:

```python
ai = Intelisys(provider="openai", model="gpt-4")
```

In this example, we're creating an Intelisys instance named `ai`, and we're telling it to talk to the "OpenAI" provider using the "gpt-4" model.

### Step 4: Chat with Your AI Friend

Now that we have our Intelisys instance, we can start chatting with our AI friend! Let's ask it a question:

```python
response = ai.chat("What is the capital of France?")
print(response)
```

This will send the message "What is the capital of France?" to our AI friend, and print out its response.

## Advanced Features

### Template-Based Chat

Imagine you want to ask your AI friend to explain something in simple terms. We can use a template to make it easier:

```python
ai = Intelisys(provider="anthropic", model="claude-3-5-sonnet-20240620")
ai.set_default_template("Explain {{topic}} in simple terms.")
response = ai.template_chat(render_data={"topic": "artificial intelligence"})
print(response)
```

In this example, we're setting a default template for our messages, and then using the `template_chat` method to send a message with the topic "artificial intelligence" filled in.

### Asynchronous Chat

Sometimes, we might want to chat with our AI friend in the background while we do other things. We can use asynchronous chat for this:

```python
import asyncio

async def async_chat():
    ai = Intelisys(provider="openai", model="gpt-4", use_async=True)
    response = await ai.chat_async("What are the implications of AGI?")
    print(response)

asyncio.run(async_chat())
```

In this example, we're creating an asynchronous function `async_chat` that creates an Intelisys instance with asynchronous mode enabled. We then use the `chat_async` method to send a message and wait for the response.

### Structured Output (OpenAI only)

Imagine you want your AI friend to give you a structured response, like a movie review with a title, rating, and summary. We can use Pydantic models for this:

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

In this example, we're defining a Pydantic model `MovieReview` with fields for the title, rating, and summary. We then set this model as the output model for our Intelisys instance, and ask it to review the movie "Inception". The response will be a `MovieReview` instance with the requested information.

### Reference Information

Imagine you want to give your AI friend some reference information to help it answer your question. We can use the `reference` method for this:

```python
ai = Intelisys(provider="openai", model="gpt-4")
ai.reference("https://example.com/article.html")
ai.reference("/path/to/local/document.pdf")
response = ai.chat("Summarize the referenced information")
print(response)
```

In this example, we're adding two references to our Intelisys instance: a URL and a local PDF file. We then ask it to summarize the referenced information, and print out the response.

## API Reference

### Intelisys Class

#### `__init__(self, name="Intelisys", provider="anthropic", model=None, ...)`
Initializes an Intelisys instance.

#### `chat(self, user_input)`
Sends a chat message and returns the AI's response.

#### `image(self, path_or_url: str)`
Adds an image to the current message for image-based AI tasks.

#### `set_system_message(self, message=None)`
Sets the system message for the conversation.

#### `set_default_template(self, template: str)`
Sets a default template for formatting messages.

#### `template_chat(self, render_data=None, template=None, persona=None)`
Sends a chat message using a template and returns the AI's response.

#### `reference(self, source: str)`
Adds content from a URL, file, or PDF to the system message.

#### `set_output_model(self, model: Type[BaseModel])`
Sets the Pydantic model for structured output (OpenAI provider only).

#### `results(self)`
Returns the results of the last chat operation.

For asynchronous versions of these methods (where applicable), append `_async` to the method name.

## Contributing

We welcome contributions to Intelisys! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

For a detailed list of changes and version history, please refer to the [CHANGELOG.md](CHANGELOG.md) file.