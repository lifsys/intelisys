import pytest
import asyncio
import json
import os
from unittest.mock import Mock, patch, AsyncMock
from intelisys.intelisys import Intelisys, safe_json_loads, iterative_llm_fix_json

@pytest.fixture
def intelisys_instance():
    return Intelisys(provider="openai", model="gpt-4", api_key="test_key")

def test_initialization(intelisys_instance):
    assert intelisys_instance.provider == "openai"
    assert intelisys_instance.model == "gpt-4"
    assert intelisys_instance._api_key == "test_key"

def test_set_log_level(intelisys_instance):
    intelisys_instance.set_log_level("DEBUG")
    assert intelisys_instance.logger.level == 10  # DEBUG level

def test_set_system_message(intelisys_instance):
    intelisys_instance.set_system_message("Test system message")
    assert intelisys_instance.system_message == "Test system message"

@patch('intelisys.intelisys.Intelisys.get_response')
def test_chat(mock_get_response, intelisys_instance):
    mock_get_response.return_value = "Test response"
    response = intelisys_instance.chat("Test input")
    assert response == "Test response"
    assert intelisys_instance.current_message == {"type": "text", "text": "Test input"}

@patch('os.path.exists')
@patch('intelisys.intelisys.Intelisys._encode_image')
def test_image(mock_encode_image, mock_exists, intelisys_instance):
    mock_exists.return_value = True
    mock_encode_image.return_value = "encoded_image_data"
    intelisys_instance.image("/path/to/image.jpg")
    assert intelisys_instance.image_urls == ["/path/to/image.jpg"]

@patch('intelisys.intelisys.Intelisys._create_response')
@patch('intelisys.intelisys.Intelisys._handle_response')
def test_get_response(mock_handle_response, mock_create_response, intelisys_instance):
    mock_create_response.return_value = "Raw response"
    mock_handle_response.return_value = "Processed response"
    intelisys_instance.current_message = {"type": "text", "text": "Test input"}
    response = intelisys_instance.get_response()
    assert response == "Processed response"
    assert intelisys_instance.last_response == "Processed response"

def test_trim_history(intelisys_instance):
    intelisys_instance.history = [
        {"role": "user", "content": "Test " * 1000},
        {"role": "assistant", "content": "Response " * 1000}
    ]
    original_length = len(intelisys_instance.history)
    intelisys_instance.trim_history()
    assert len(intelisys_instance.history) <= original_length

def test_add_message(intelisys_instance):
    intelisys_instance.add_message("user", "Test message")
    assert intelisys_instance.history[-1] == {"role": "user", "content": "Test message"}

def test_set_default_template(intelisys_instance):
    intelisys_instance.set_default_template("Hello, {{name}}")
    assert intelisys_instance.default_template == "Hello, {{name}}"

def test_set_default_persona(intelisys_instance):
    intelisys_instance.set_default_persona("You are a helpful assistant")
    assert intelisys_instance.default_persona == "You are a helpful assistant"

@patch('intelisys.intelisys.Intelisys._go_get_api')
def test_set_template_instruction(mock_go_get_api, intelisys_instance):
    mock_go_get_api.return_value = "Test instruction"
    intelisys_instance.set_template_instruction("test_set", "test_instruction")
    assert intelisys_instance.template_instruction == "Test instruction"

@patch('intelisys.intelisys.Intelisys._go_get_api')
def test_set_template_persona(mock_go_get_api, intelisys_instance):
    mock_go_get_api.return_value = "Test persona"
    intelisys_instance.set_template_persona("test_persona")
    assert intelisys_instance.template_persona == "Test persona"

def test_set_template_data(intelisys_instance):
    test_data = {"name": "Alice", "age": 30}
    intelisys_instance.set_template_data(test_data)
    assert intelisys_instance.template_data == test_data

@patch('intelisys.intelisys.Intelisys.chat')
def test_template_chat(mock_chat, intelisys_instance):
    mock_chat.return_value = "Test response"
    response = intelisys_instance.template_chat(
        render_data={"name": "Alice"},
        template="Hello, {{name}}",
        persona="You are a friendly assistant"
    )
    assert response == "Test response"
    mock_chat.assert_called_with("Hello, Alice")

def test_template_context(intelisys_instance):
    with intelisys_instance.template_context(template="Test {{var}}", persona="Test persona"):
        assert intelisys_instance.default_template == "Test {{var}}"
        assert intelisys_instance.default_persona == "Test persona"
    assert intelisys_instance.default_template != "Test {{var}}"
    assert intelisys_instance.default_persona != "Test persona"

@pytest.mark.asyncio
async def test_chat_async(intelisys_instance):
    with patch.object(intelisys_instance, 'get_response_async', new_callable=AsyncMock) as mock_get_response:
        mock_get_response.return_value = "Async response"
        response = await intelisys_instance.chat_async("Async test input")
        assert response == "Async response"
        assert intelisys_instance.history[-1]["role"] == "user"
        assert intelisys_instance.history[-1]["content"] == "Async test input"

@pytest.mark.asyncio
async def test_add_message_async(intelisys_instance):
    await intelisys_instance.add_message_async("user", "Async test message")
    assert intelisys_instance.history[-1] == {"role": "user", "content": "Async test message"}

@pytest.mark.asyncio
async def test_set_system_message_async(intelisys_instance):
    await intelisys_instance.set_system_message_async("Async system message")
    assert intelisys_instance.system_message == "Async system message"

@pytest.mark.asyncio
async def test_get_response_async(intelisys_instance):
    with patch.object(intelisys_instance, '_create_response_async', new_callable=AsyncMock) as mock_create_response:
        mock_create_response.return_value = AsyncMock()
        mock_create_response.return_value.__aiter__.return_value = [
            AsyncMock(choices=[AsyncMock(delta=AsyncMock(content="Async "))])
        ]
        mock_create_response.return_value.choices = [AsyncMock(message=AsyncMock(content="Async "))]
        response = await intelisys_instance.get_response_async()
        assert response == "Async "
        assert intelisys_instance.history[-1]["role"] == "assistant"
        assert intelisys_instance.history[-1]["content"] == "Async "

@pytest.mark.asyncio
async def test_template_chat_async(intelisys_instance):
    with patch.object(intelisys_instance, 'chat_async', new_callable=AsyncMock) as mock_chat_async:
        mock_chat_async.return_value = "Async template response"
        result = await intelisys_instance.template_chat_async(
            render_data={"name": "Bob"},
            template="Hello, {{name}}",
            persona="You are an async assistant"
        )
        assert result.last_response == "Async template response"

def test_safe_json_loads():
    # Test valid JSON
    assert safe_json_loads('{"key": "value"}') == {"key": "value"}
    
    # Test invalid JSON that can be fixed
    assert safe_json_loads("{'key': 'value'}") == {"key": "value"}
    
    # Test completely invalid input
    result = safe_json_loads("This is not JSON at all")
    assert isinstance(result, dict)
    assert "content" in result

@patch('intelisys.intelisys.Intelisys')
def test_iterative_llm_fix_json(mock_intelisys):
    mock_instance = Mock()
    mock_instance.chat.return_value = mock_instance
    mock_instance.get_response.return_value = '{"fixed": "json"}'
    
    result = iterative_llm_fix_json('{"broken": "json"}', intelisys_instance=mock_instance)
    print(f"Test result: {result}")
    assert result == '{"fixed": "json"}'
    
    # Verify that the mock was called correctly
    mock_instance.chat.assert_called_once_with('Fix this JSON: {"broken": "json"}')
    mock_instance.get_response.assert_called_once()

if __name__ == "__main__":
    pytest.main()