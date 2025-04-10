#!/usr/bin/env python3
"""
Simple script to test API keys using direct API calls.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_api():
    """Test OpenAI API key using direct API call."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OpenAI API key")
        return False
    
    print(f"Testing OpenAI API key: {api_key[:10]}...")
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
        )
        
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_anthropic_api():
    """Test Anthropic API key using direct API call."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Missing Anthropic API key")
        return False
    
    print(f"Testing Anthropic API key: {api_key[:10]}...")
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-opus-20240229",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing API keys...")
    
    openai_success = test_openai_api()
    print(f"OpenAI API test: {'Success' if openai_success else 'Failed'}")
    
    anthropic_success = test_anthropic_api()
    print(f"Anthropic API test: {'Success' if anthropic_success else 'Failed'}")
    
    if openai_success and anthropic_success:
        print("All API keys are working!")
    else:
        print("Some API keys are not working. Please check the errors above.") 