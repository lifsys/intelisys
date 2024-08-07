import pytest
import json
from unittest.mock import patch, MagicMock
from intelisys import Intelisys, safe_json_loads

@pytest.fixture
def intelisys_instance():
    return Intelisys(provider="openai", model="gpt-4o-mini", json_mode=True)

def test_initialization(intelisys_instance):
    assert isinstance(intelisys_instance, Intelisys)
    assert intelisys_instance.provider == "openai"
    assert intelisys_instance.model == "gpt-4o-mini"
    assert intelisys_instance.json_mode == True

def test_set_system_message(intelisys_instance):
    test_message = "You are a test assistant."
    intelisys_instance.set_system_message(test_message)
    assert intelisys_instance.system_message == test_message + " Please return your response in JSON unless user has specified a system message."

@patch('intelisys.OpenAI')
def test_chat_and_send(mock_openai, intelisys_instance):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({"response": "Test response"})
    mock_client.chat.completions.create.return_value = mock_response

    intelisys_instance.chat("Test input").send()
    
    assert len(intelisys_instance.history) == 2  # System message and user input
    assert intelisys_instance.history[-2]['role'] == 'user'
    assert intelisys_instance.history[-2]['content'] == 'Test input'
    assert intelisys_instance.history[-1]['role'] == 'assistant'
    assert json.loads(intelisys_instance.history[-1]['content']) == {"response": "Test response"}

def test_image_method(intelisys_instance):
    with pytest.raises(ValueError):
        intelisys_instance.image("test_image.jpg")  # Should raise error for non-OpenAI/OpenRouter providers

    intelisys_instance.provider = "openai"
    intelisys_instance.image("https://example.com/image.jpg")
    assert len(intelisys_instance.image_urls) == 1
    assert intelisys_instance.image_urls[0]['type'] == 'image_url'
    assert intelisys_instance.image_urls[0]['image_url']['url'] == 'https://example.com/image.jpg'

def test_safe_json_loads():
    valid_json = '{"key": "value"}'
    assert safe_json_loads(valid_json) == {"key": "value"}

    invalid_json = '{"key": "value",}'
    with pytest.raises(ValueError):
        safe_json_loads(invalid_json)

@patch('intelisys.Intelisys')
def test_iterative_llm_fix_json(mock_intelisys):
    mock_instance = MagicMock()
    mock_intelisys.return_value = mock_instance
    mock_instance.results.return_value = '{"fixed": "json"}'

    result = intelisys.iterative_llm_fix_json('{"broken: "json"}')
    assert result == '{"fixed": "json"}'

@pytest.mark.asyncio
async def test_async_methods(intelisys_instance):
    with patch('intelisys.AsyncOpenAI') as mock_async_openai:
        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"response": "Async test response"})
        mock_client.chat.completions.create.return_value = mock_response

        await intelisys_instance.chat_async("Async test input")
        
        assert len(intelisys_instance.history) == 2  # System message and user input
        assert intelisys_instance.history[-2]['role'] == 'user'
        assert intelisys_instance.history[-2]['content'] == 'Async test input'
        assert intelisys_instance.history[-1]['role'] == 'assistant'
        assert json.loads(intelisys_instance.history[-1]['content']) == {"response": "Async test response"}

def test_template_chat(intelisys_instance):
    intelisys_instance.set_default_template("Hello, {{ name }}!")
    intelisys_instance.set_default_persona("You are a greeting bot.")

    with patch.object(intelisys_instance, 'chat') as mock_chat, \
         patch.object(intelisys_instance, 'send') as mock_send, \
         patch.object(intelisys_instance, 'results') as mock_results:
        
        mock_results.return_value = {"greeting": "Hello, Alice!"}
        
        result = intelisys_instance.template_chat({"name": "Alice"}, return_self=False)
        
        mock_chat.assert_called_once_with("Hello, Alice!")
        mock_send.assert_called_once()
        mock_results.assert_called_once()
        
        assert result == {"greeting": "Hello, Alice!"}

if __name__ == "__main__":
    pytest.main()
