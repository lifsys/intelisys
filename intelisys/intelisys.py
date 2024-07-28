"""
Provides intelligence/AI services for the Lifsys Enterprise.

This module requires a 1Password Connect server to be available and configured.
The OP_CONNECT_TOKEN and OP_CONNECT_HOST environment variables must be set
for the onepasswordconnectsdk to function properly.
"""
import os
import json
from time import sleep
from typing import Optional, Dict, List
from openai import OpenAI
from litellm import completion
from jinja2 import Template
from onepasswordconnectsdk import new_client_from_environment

def get_api(item: str, key_name: str, vault: str = "API") -> str:
    """
    Retrieve an API key from 1Password.

    This function connects to a 1Password vault using the onepasswordconnectsdk
    and retrieves a specific API key. It requires the OP_CONNECT_TOKEN and
    OP_CONNECT_HOST environment variables to be set for proper functionality.

    Args:
        item (str): The name of the item in 1Password containing the API key.
        key_name (str): The specific field name within the item that holds the API key.
        vault (str, optional): The name of the 1Password vault to search in. Defaults to "API".

    Returns:
        str: The retrieved API key as a string.

    Raises:
        Exception: If there's an error connecting to 1Password or retrieving the key.
            The exception message will include details about the specific error encountered.

    Example:
        >>> api_key = get_api("OpenAI", "API_KEY", "Development")
        >>> print(api_key)
        'sk-1234567890abcdef1234567890abcdef'
    """
    try:
        client = new_client_from_environment()
        item = client.get_item(item, vault)
        for field in item.fields:
            if field.label == key_name:
                return field.value
    except Exception as e:
        raise Exception(f"Connect Error: {e}")


def template_api_json(model: str, render_data: Dict, system_message: str, persona: str) -> Dict:
    """
    Get the completion response from the API using the specified model and return it as a JSON object.

    This function sends a request to an AI model API, using a templated system message
    and specified persona. It then processes the response to ensure it's in valid JSON format.

    Args:
        model (str): The name of the AI model to use for the API call (e.g., "gpt-4", "claude-3.5").
        render_data (Dict): A dictionary containing data to render the template. 
            For example: {"name": "John", "age": 30}.
        system_message (str): A Jinja2 template string to be used as the system message.
            This will be rendered with the render_data.
        persona (str): A string describing the persona or role the AI should adopt for this response.

    Returns:
        Dict: The API response parsed as a Python dictionary. The structure of this dictionary
        will depend on the specific response from the AI model.

    Raises:
        json.JSONDecodeError: If the API response cannot be parsed as valid JSON.
        Any exceptions raised by the underlying get_completion_api or json.loads functions.

    Example:
        >>> model = "gpt-4"
        >>> render_data = {"user_name": "Alice", "task": "summarize"}
        >>> system_message = "You are an AI assistant helping {{user_name}} to {{task}} a document."
        >>> persona = "helpful assistant"
        >>> response = template_api_json(model, render_data, system_message, persona)
        >>> print(response)
        {'summary': 'This is a summary of the document...', 'key_points': ['Point 1', 'Point 2']}
    """
    xtemplate = Template(system_message)
    prompt = xtemplate.render(render_data)
    response = get_completion_api(prompt, model, "system", persona)
    response = response.strip("```json\n").strip("```").strip()
    response = json.loads(response)
    return response

def template_api(model: str, render_data: Dict, system_message: str, persona: str) -> str:
    """
    Get the completion response from the API using the specified model.

    This function is similar to template_api_json, but returns the raw string response
    from the AI model instead of attempting to parse it as JSON.

    Args:
        model (str): The name of the AI model to use for the API call (e.g., "gpt-4", "claude-3.5").
        render_data (Dict): A dictionary containing data to render the template. 
            For example: {"name": "John", "age": 30}.
        system_message (str): A Jinja2 template string to be used as the system message.
            This will be rendered with the render_data.
        persona (str): A string describing the persona or role the AI should adopt for this response.

    Returns:
        str: The raw API response as a string. This could be in any format, depending on
        the AI model's output (e.g., plain text, markdown, or even JSON as a string).

    Raises:
        Any exceptions raised by the underlying get_completion_api function.

    Example:
        >>> model = "gpt-4"
        >>> render_data = {"topic": "artificial intelligence"}
        >>> system_message = "Explain {{topic}} in simple terms."
        >>> persona = "friendly teacher"
        >>> response = template_api(model, render_data, system_message, persona)
        >>> print(response)
        "Artificial Intelligence, or AI, is like teaching computers to think and learn..."
    """
    xtemplate = Template(system_message)
    prompt = xtemplate.render(render_data)
    response = get_completion_api(prompt, model, "system", persona)
    return response

def initialize_client() -> OpenAI:
    """
    Initialize the OpenAI client with the API key retrieved from 1Password.

    This function retrieves the OpenAI API key from 1Password using the get_api function,
    then initializes and returns an OpenAI client instance.

    Returns:
        OpenAI: An initialized OpenAI client instance ready for making API calls.

    Raises:
        Exception: If there's an error retrieving the API key or initializing the client.
            This could be due to issues with 1Password access or invalid API keys.

    Example:
        >>> client = initialize_client()
        >>> # Now you can use the client to make OpenAI API calls
        >>> response = client.completions.create(model="text-davinci-002", prompt="Hello, AI!")
    """
    api_key = get_api("OPEN-AI", "Mamba")
    return OpenAI(api_key=api_key)

def create_thread(client: OpenAI):
    """
    Create a new thread using the OpenAI client.

    This function initializes a new conversation thread using the provided OpenAI client.
    Threads are used to maintain context in multi-turn conversations with AI assistants.

    Args:
        client (OpenAI): An initialized OpenAI client object.

    Returns:
        Thread: A new thread object representing the created conversation thread.

    Example:
        >>> client = initialize_client()
        >>> new_thread = create_thread(client)
        >>> print(new_thread.id)
        'thread_abc123...'
    """
    return client.beta.threads.create()

def send_message(client: OpenAI, thread_id: str, reference: str):
    """
    Send a message to the specified thread.

    This function adds a new message to an existing conversation thread. The message
    is sent with the 'user' role, representing input from the user to the AI assistant.

    Args:
        client (OpenAI): An initialized OpenAI client object.
        thread_id (str): The ID of the thread to send the message to.
        reference (str): The content of the message to send.

    Returns:
        Message: The created message object, containing details about the sent message.

    Example:
        >>> client = initialize_client()
        >>> thread_id = "thread_abc123..."
        >>> message = send_message(client, thread_id, "What's the weather like today?")
        >>> print(message.content)
        'What's the weather like today?'
    """
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=reference,
    )

def run_assistant(client: OpenAI, thread_id: str, assistant_id: str):
    """
    Run the assistant for the specified thread.

    This function initiates a new run of the AI assistant on the specified thread.
    A run represents the assistant's process of analyzing the conversation and generating a response.

    Args:
        client (OpenAI): An initialized OpenAI client object.
        thread_id (str): The ID of the thread to run the assistant on.
        assistant_id (str): The ID of the assistant to run.

    Returns:
        Run: The created run object, containing details about the initiated assistant run.

    Example:
        >>> client = initialize_client()
        >>> thread_id = "thread_abc123..."
        >>> assistant_id = "asst_def456..."
        >>> run = run_assistant(client, thread_id, assistant_id)
        >>> print(run.status)
        'queued'
    """
    return client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=assistant_id
    )

def wait_for_run_completion(client: OpenAI, thread_id: str, run_id: str):
    """
    Wait for the assistant run to complete.

    This function polls the status of a run until it's no longer in a 'queued' or 'in_progress' state.
    It includes a small delay between checks to avoid excessive API calls.

    Args:
        client (OpenAI): An initialized OpenAI client object.
        thread_id (str): The ID of the thread.
        run_id (str): The ID of the run to wait for.

    Returns:
        Run: The completed run object, containing the final status and any output from the assistant.

    Example:
        >>> client = initialize_client()
        >>> thread_id = "thread_abc123..."
        >>> run_id = "run_ghi789..."
        >>> completed_run = wait_for_run_completion(client, thread_id, run_id)
        >>> print(completed_run.status)
        'completed'
    """
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    while run.status in ["queued", "in_progress"]:
        sleep(0.5)  # Add a delay to avoid rapid polling
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    return run

def get_assistant_responses(client: OpenAI, thread_id: str) -> List[str]:
    """
    Retrieve and clean the assistant's responses from the thread.

    This function fetches all messages in a thread and extracts the content of messages
    sent by the assistant. It returns these responses as a list of strings.

    Args:
        client (OpenAI): An initialized OpenAI client object.
        thread_id (str): The ID of the thread to retrieve responses from.

    Returns:
        List[str]: A list of assistant responses, where each response is a string.

    Example:
        >>> client = initialize_client()
        >>> thread_id = "thread_abc123..."
        >>> responses = get_assistant_responses(client, thread_id)
        >>> for response in responses:
        ...     print(response)
        'Here's the weather forecast for today...'
        'Is there anything else you'd like to know?'
    """
    message_list = client.beta.threads.messages.list(thread_id=thread_id)
    assistant_responses = [
        message.content[0].text.value
        for message in message_list.data
        if message.role == "assistant"
    ]
    return assistant_responses

def get_assistant(reference: str, assistant_id: str) -> List[str]:
    """
    Get the assistant's response for the given reference and assistant ID.

    This function encapsulates the entire process of interacting with an AI assistant:
    creating a thread, sending a message, running the assistant, and retrieving the responses.

    Args:
        reference (str): The reference message to send to the assistant. This is typically
                         the user's question or prompt.
        assistant_id (str): The ID of the assistant to use for generating the response.

    Returns:
        List[str]: A list of assistant responses. Each response is a string containing
                   the assistant's reply to the reference message.

    Example:
        >>> reference = "What are the three laws of robotics?"
        >>> assistant_id = "asst_def456..."
        >>> responses = get_assistant(reference, assistant_id)
        >>> for response in responses:
        ...     print(response)
        '1. A robot may not injure a human being or, through inaction, allow a human being to come to harm...'
    """
    client = initialize_client()
    thread = create_thread(client)
    send_message(client, thread.id, reference)
    run = run_assistant(client, thread.id, assistant_id)
    wait_for_run_completion(client, thread.id, run.id)
    responses = get_assistant_responses(client, thread.id)
    return responses

def get_completion_api(
    prompt: str,
    model_name: str,
    mode: str = "simple",
    system_message: Optional[str] = None
) -> Optional[str]:
    """
    Get the completion response from the API using the specified model.

    Args:
        prompt (str): The prompt to send to the API.
        model_name (str): The name of the model to use for completion. Supported models include:
            'gpt-4o-mini', 'gpt-4', 'gpt-4o', 'claude-3.5', 'gemini-flash', 'llama-3-70b',
            'llama-3.1-large', 'groq-llama', 'groq-fast', 'mistral-large'.
        mode (str, optional): The mode of message sending ('simple' or 'system'). Defaults to "simple".
        system_message (Optional[str], optional): The system message to send if in system mode. Defaults to None.

    Returns:
        Optional[str]: The completion response content, or None if an error occurs.

    Raises:
        ValueError: If an unsupported model or mode is specified.
    """
    try:
        # Model configurations
        model_configs = {
            "gpt-4o-mini": ("OPEN-AI", "Mamba", lambda x: x),
            "gpt-4": ("OPEN-AI", "Mamba", lambda x: x),
            "gpt-4o": ("OPEN-AI", "Mamba", lambda x: x),
            "claude-3.5": ("Anthropic", "CLI-Maya", lambda _: "claude-3-5-sonnet-20240620"),
            "gemini-flash": ("Gemini", "CLI-Maya", lambda _: "gemini/gemini-1.5-flash"),
            "llama-3-70b": ("TogetherAI", "API", lambda _: "together_ai/meta-llama/Llama-3-70b-chat-hf"),
            "llama-3.1-large": ("TogetherAI", "API", lambda _: "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"),
            "groq-llama": ("Groq", "Promptsys", lambda _: "groq/llama3-70b-8192"),
            "groq-fast": ("Groq", "Promptsys", lambda _: "groq/llama3-8b-8192"),
            "mistral-large": ("MistralAI", "API", lambda _: "mistral/mistral-large-latest"),
        }

        if model_name not in model_configs:
            raise ValueError(f"Unsupported model: {model_name}")

        api_name, key_name, model_func = model_configs[model_name]
        os.environ[f"{api_name.upper()}_API_KEY"] = get_api(api_name, key_name)
        selected_model = model_func(model_name)

        # Select message type
        match mode:
            case "simple":
                print("Message Simple")
                messages = [{"content": prompt, "role": "user"}]
            case "system":
                if system_message is None:
                    raise ValueError("system_message must be provided in system mode")
                messages = [
                    {"content": system_message, "role": "system"},
                    {"content": prompt, "role": "user"},
                ]
            case _:
                raise ValueError(f"Unsupported mode: {mode}")

        # Make the API call
        response = completion(
            model=selected_model,
            messages=messages,
            temperature=0.1,
        )

        # Extract and return the response content
        return response["choices"][0]["message"]["content"]

    except KeyError as ke:
        print(f"Key error occurred: {ke}")
    except ValueError as ve:
        print(f"Value error occurred: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None

def fix_json(json_string: str) -> str:
    """
    Fix and format a potentially malformed JSON string.

    This function uses an AI model to correct and format a given JSON string.
    It's particularly useful for handling JSON that may have syntax errors or
    formatting issues.

    Args:
        json_string (str): The JSON string to fix and format. This can be a
            malformed or incorrectly formatted JSON string.

    Returns:
        str: The fixed and formatted JSON string. If the input is valid JSON,
            it will be returned in a standardized format. If the input is invalid,
            the function attempts to correct common errors and return a valid JSON string.

    Example:
        >>> malformed_json = "{'key': 'value', 'nested': {'a':1, 'b': 2,}}"
        >>> fixed_json = fix_json(malformed_json)
        >>> print(fixed_json)
        '{"key": "value", "nested": {"a": 1, "b": 2}}'
    """
    prompt = f"You are a JSON formatter, fixing any issues with JSON formats. Review the following JSON: {json_string}. Return a fixed JSON formatted string but do not lead with ```json\n, without making changes to the content."
    return get_completion_api(prompt, "gemini-flash", "system", prompt)
