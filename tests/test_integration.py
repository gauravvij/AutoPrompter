"""
Integration tests for LocalLLMClient and config loading.
Uses mocked responses to verify functionality without requiring actual local servers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch, MagicMock
import json
from src.local_llm_client import LocalLLMClient, LLMResponse
from src.config_manager import Config, LocalLLMConfig, LLMConfig


def test_ollama_client_query():
    """Test Ollama client with mocked response."""
    print("\n=== Testing Ollama Client Query ===")
    
    config = LocalLLMConfig(
        backend="ollama",
        model="llama3.2",
        host="http://localhost",
        port=11434
    )
    
    client = LocalLLMClient(config)
    
    # Mock the session.post response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": "Hello, this is a test response."},
        "prompt_eval_count": 10,
        "eval_count": 20
    }
    
    with patch.object(client.session, 'post', return_value=mock_response):
        response = client.query("Say hello")
        
        assert response.success, f"Expected success, got error: {response.error}"
        assert "Hello" in response.content, f"Expected 'Hello' in content, got: {response.content}"
        assert response.usage['completion_tokens'] == 20
        print(f"✓ Ollama query successful: {response.content[:50]}...")
        print(f"  Latency: {response.latency_ms:.2f}ms")
        print(f"  Usage: {response.usage}")


def test_llama_cpp_client_query():
    """Test llama.cpp client with mocked response."""
    print("\n=== Testing llama.cpp Client Query ===")
    
    config = LocalLLMConfig(
        backend="llama_cpp",
        model="llama-3.2-3b",
        host="http://localhost",
        port=8080
    )
    
    client = LocalLLMClient(config)
    
    # Mock the session.post response (OpenAI-compatible format)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama-3.2-3b",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a llama.cpp test response."
                }
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40
        }
    }
    
    with patch.object(client.session, 'post', return_value=mock_response):
        response = client.query("Test prompt")
        
        assert response.success, f"Expected success, got error: {response.error}"
        assert "llama.cpp" in response.content, f"Expected 'llama.cpp' in content, got: {response.content}"
        assert response.usage['completion_tokens'] == 25
        print(f"✓ llama.cpp query successful: {response.content[:50]}...")
        print(f"  Latency: {response.latency_ms:.2f}ms")
        print(f"  Usage: {response.usage}")


def test_query_with_history():
    """Test conversation with history."""
    print("\n=== Testing Query with History ===")
    
    config = LocalLLMConfig(
        backend="ollama",
        model="llama3.2",
        host="http://localhost",
        port=11434
    )
    
    client = LocalLLMClient(config)
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": "The capital of France is Paris."},
        "prompt_eval_count": 50,
        "eval_count": 15
    }
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    with patch.object(client.session, 'post', return_value=mock_response):
        response = client.query_with_history(messages)
        
        assert response.success
        assert "Paris" in response.content
        print(f"✓ Query with history successful: {response.content}")


def test_batch_query():
    """Test batch query processing."""
    print("\n=== Testing Batch Query ===")
    
    config = LocalLLMConfig(
        backend="llama_cpp",
        model="llama-3.2-3b",
        host="http://localhost",
        port=8080
    )
    
    client = LocalLLMClient(config)
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "llama-3.2-3b",
        "choices": [{"message": {"role": "assistant", "content": "Response"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    }
    
    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    
    with patch.object(client.session, 'post', return_value=mock_response):
        responses = client.batch_query(prompts)
        
        assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}"
        assert all(r.success for r in responses), "All responses should be successful"
        print(f"✓ Batch query successful: {len(responses)} responses processed")


def test_config_loading_ollama():
    """Test loading Ollama configuration from YAML."""
    print("\n=== Testing Ollama Config Loading ===")
    
    config = Config.from_yaml("/root/AutoPrompter/config_ollama.yaml")
    
    assert isinstance(config.optimizer_llm, LocalLLMConfig), "Optimizer should use LocalLLMConfig"
    assert config.optimizer_llm.backend == "ollama"
    assert config.optimizer_llm.model == "llama3.2"
    assert config.optimizer_llm.port == 11434
    
    errors = config.validate()
    assert len(errors) == 0, f"Config validation failed: {errors}"
    
    print(f"✓ Ollama config loaded successfully")
    print(f"  Optimizer: {config.optimizer_llm.model} (port {config.optimizer_llm.port})")
    print(f"  Target: {config.target_llm.model} (port {config.target_llm.port})")


def test_config_loading_llama_cpp():
    """Test loading llama.cpp configuration from YAML."""
    print("\n=== Testing llama.cpp Config Loading ===")
    
    config = Config.from_yaml("/root/AutoPrompter/config_llama_cpp.yaml")
    
    assert isinstance(config.optimizer_llm, LocalLLMConfig), "Optimizer should use LocalLLMConfig"
    assert config.optimizer_llm.backend == "llama_cpp"
    assert config.optimizer_llm.model == "llama-3.2-3b"
    assert config.optimizer_llm.port == 8080
    
    errors = config.validate()
    assert len(errors) == 0, f"Config validation failed: {errors}"
    
    print(f"✓ llama.cpp config loaded successfully")
    print(f"  Optimizer: {config.optimizer_llm.model} (port {config.optimizer_llm.port})")
    print(f"  Target: {config.target_llm.model} (port {config.target_llm.port})")


def test_openrouter_backward_compatibility():
    """Test that OpenRouter configs still work (backward compatibility)."""
    print("\n=== Testing OpenRouter Backward Compatibility ===")
    
    config = Config.from_yaml("/root/AutoPrompter/config.yaml")
    
    # Should use LLMConfig for OpenRouter backend
    assert isinstance(config.optimizer_llm, LLMConfig), "Optimizer should use LLMConfig for OpenRouter"
    assert config.optimizer_llm.backend == "openrouter"
    assert config.optimizer_llm.api_base == "https://openrouter.ai/api/v1"
    
    errors = config.validate()
    assert len(errors) == 0, f"Config validation failed: {errors}"
    
    print(f"✓ OpenRouter config loaded successfully (backward compatible)")
    print(f"  Optimizer: {config.optimizer_llm.model}")
    print(f"  Target: {config.target_llm.model}")


def test_connection_check():
    """Test connection check method (should fail gracefully without server)."""
    print("\n=== Testing Connection Check ===")
    
    config = LocalLLMConfig(
        backend="ollama",
        model="llama3.2",
        host="http://localhost",
        port=11434
    )
    
    client = LocalLLMClient(config)
    
    # This should return False since no server is running
    is_connected = client.check_connection()
    print(f"✓ Connection check completed (connected: {is_connected})")
    print(f"  (Expected False since no Ollama server is running)")


def test_list_models():
    """Test list models method (should fail gracefully without server)."""
    print("\n=== Testing List Models ===")
    
    config = LocalLLMConfig(
        backend="ollama",
        model="llama3.2",
        host="http://localhost",
        port=11434
    )
    
    client = LocalLLMClient(config)
    
    # This should return empty list since no server is running
    models = client.list_models()
    print(f"✓ List models completed (models: {models})")
    print(f"  (Expected empty list since no Ollama server is running)")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("INTEGRATION TESTS FOR LOCAL LLM CLIENT")
    print("=" * 60)
    
    tests = [
        test_ollama_client_query,
        test_llama_cpp_client_query,
        test_query_with_history,
        test_batch_query,
        test_config_loading_ollama,
        test_config_loading_llama_cpp,
        test_openrouter_backward_compatibility,
        test_connection_check,
        test_list_models,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
