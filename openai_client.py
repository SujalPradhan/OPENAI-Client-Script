#!/usr/bin/env python3
"""
Reusable OpenAI Chat Completions Script
A comprehensive, production-ready wrapper for OpenAI's Chat Completions API.

Usage:
    # As CLI tool
    python openai_chat.py "What is machine learning?"
    
    # As importable module
    from openai_chat import ChatClient, get_completion
    
Author: Assistant
License: MIT
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import List, Dict, Optional, Union, Any, Generator
from dataclasses import dataclass, asdict
import openai
from openai import OpenAI
import backoff
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    """Configuration class for chat parameters."""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False
    log_requests: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ChatConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


@dataclass
class ChatResponse:
    """Response object containing completion data and metadata."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float
    raw_response: Dict[str, Any]


class MessageManager:
    """Helper class for managing conversation messages."""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
    
    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self.messages.append({"role": "system", "content": content})
    
    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.messages.append({"role": "assistant", "content": content})
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message with specified role."""
        if role not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role: {role}. Must be 'system', 'user', or 'assistant'")
        self.messages.append({"role": role, "content": content})
    
    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages.clear()
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get copy of current messages."""
        return self.messages.copy()
    
    def load_conversation(self, messages: List[Dict[str, str]]) -> None:
        """Load a conversation history."""
        self.messages = messages.copy()
    
    def to_json(self) -> str:
        """Export messages to JSON string."""
        return json.dumps(self.messages, indent=2)
    
    def from_json(self, json_str: str) -> None:
        """Load messages from JSON string."""
        self.messages = json.loads(json_str)


class ChatClient:
    """Main chat client for OpenAI API interactions."""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[ChatConfig] = None):
        """
        Initialize the chat client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            config: ChatConfig object with default parameters
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.config = config or ChatConfig()
        self.message_manager = MessageManager()
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
        max_tries=3,
        max_time=60
    )
    def _make_api_call(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Make API call with retry logic."""
        return self.client.chat.completions.create(
            messages=messages,
            **kwargs
        )
    
    def get_completion(
        self,
        messages: Union[List[Dict[str, str]], str],
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Union[ChatResponse, Generator[str, None, None]]:
        """
        Get completion from OpenAI API.
        
        Args:
            messages: List of message dicts or single string (treated as user message)
            model: Model name (overrides config default)
            stream: Enable streaming (overrides config default)
            **kwargs: Additional parameters to override config defaults
            
        Returns:
            ChatResponse object or generator for streaming responses
        """
        # Handle string input as user message
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Merge config with overrides
        params = asdict(self.config)
        params.update(kwargs)
        if model:
            params['model'] = model
        if stream is not None:
            params['stream'] = stream
        
        # Remove non-API parameters
        log_requests = params.pop('log_requests', False)
        
        if log_requests:
            logger.info(f"Making API request with model: {params['model']}")
            logger.debug(f"Messages: {json.dumps(messages, indent=2)}")
            logger.debug(f"Parameters: {json.dumps(params, indent=2)}")
        
        start_time = time.time()
        
        try:
            if params.get('stream', False):
                return self._handle_streaming_response(messages, params, log_requests)
            else:
                response = self._make_api_call(messages, **params)
                return self._create_response_object(response, start_time, log_requests)
                
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    
    def _handle_streaming_response(
        self, 
        messages: List[Dict[str, str]], 
        params: Dict[str, Any], 
        log_requests: bool
    ) -> Generator[str, None, None]:
        """Handle streaming response."""
        try:
            stream = self._make_api_call(messages, **params)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    if log_requests:
                        logger.debug(f"Streaming chunk: {content}")
                    yield content
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            raise
    
    def _create_response_object(
        self, 
        response: Any, 
        start_time: float, 
        log_requests: bool
    ) -> ChatResponse:
        """Create ChatResponse object from API response."""
        response_time = time.time() - start_time
        
        chat_response = ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=response.usage.model_dump(),
            finish_reason=response.choices[0].finish_reason,
            response_time=response_time,
            raw_response=response.model_dump()
        )
        
        if log_requests:
            logger.info(f"Response received in {response_time:.2f}s")
            logger.info(f"Token usage: {chat_response.usage}")
            logger.debug(f"Response content: {chat_response.content}")
        
        return chat_response
    
    def chat_interactive(self, system_prompt: Optional[str] = None) -> None:
        """Start an interactive chat session."""
        print("🤖 Interactive Chat (type 'quit' to exit, 'clear' to clear history)")
        print("=" * 60)
        
        if system_prompt:
            self.message_manager.add_system_message(system_prompt)
            print(f"System: {system_prompt}\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                elif user_input.lower() == 'clear':
                    self.message_manager.clear_messages()
                    if system_prompt:
                        self.message_manager.add_system_message(system_prompt)
                    print("🧹 Conversation history cleared.")
                    continue
                elif not user_input:
                    continue
                
                self.message_manager.add_user_message(user_input)
                
                if self.config.stream:
                    print("\nAssistant: ", end="", flush=True)
                    full_response = ""
                    for chunk in self.get_completion(self.message_manager.get_messages()):
                        print(chunk, end="", flush=True)
                        full_response += chunk
                    print()  # New line after streaming
                    self.message_manager.add_assistant_message(full_response)
                else:
                    response = self.get_completion(self.message_manager.get_messages())
                    print(f"\nAssistant: {response.content}")
                    self.message_manager.add_assistant_message(response.content)
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def get_completion(
    messages: Union[List[Dict[str, str]], str],
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    **kwargs
) -> ChatResponse:
    """
    Convenience function for quick completions.
    
    Args:
        messages: Messages list or single string
        model: Model name
        api_key: API key (defaults to env var)
        **kwargs: Additional chat parameters
        
    Returns:
        ChatResponse object
    """
    config = ChatConfig(model=model, **kwargs)
    client = ChatClient(api_key=api_key, config=config)
    return client.get_completion(messages)


def load_config_file(config_path: str) -> ChatConfig:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return ChatConfig.from_dict(config_dict)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return ChatConfig()
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return ChatConfig()


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(
        description="OpenAI Chat Completions CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s "What is machine learning?"
    %(prog)s "Explain Python" --model gpt-3.5-turbo --temperature 0.5
    %(prog)s --interactive --system "You are a helpful coding assistant"
    %(prog)s --config config.json "Write a poem"
        """
    )
    
    # Input options
    parser.add_argument('prompt', nargs='?', help='Prompt to send to the model')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Start interactive chat session')
    parser.add_argument('--system', '-s', help='System prompt')
    
    # Model parameters
    parser.add_argument('--model', '-m', default='gpt-4o', 
                       help='Model name (default: gpt-4o)')
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                       help='Temperature (0.0-2.0, default: 0.7)')
    parser.add_argument('--max-tokens', type=int, 
                       help='Maximum tokens in response')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-p sampling (default: 1.0)')
    parser.add_argument('--frequency-penalty', type=float, default=0.0,
                       help='Frequency penalty (-2.0 to 2.0, default: 0.0)')
    parser.add_argument('--presence-penalty', type=float, default=0.0,
                       help='Presence penalty (-2.0 to 2.0, default: 0.0)')
    
    # Options
    parser.add_argument('--stream', action='store_true',
                       help='Enable streaming response')
    parser.add_argument('--config', '-c', help='Path to JSON config file')
    parser.add_argument('--log-requests', action='store_true',
                       help='Log API requests and responses')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        config = load_config_file(args.config)
    else:
        config = ChatConfig()
    
    # Override config with command line arguments
    config.model = args.model
    config.temperature = args.temperature
    config.max_tokens = args.max_tokens
    config.top_p = args.top_p
    config.frequency_penalty = args.frequency_penalty
    config.presence_penalty = args.presence_penalty
    config.stream = args.stream
    config.log_requests = args.log_requests
    
    try:
        client = ChatClient(config=config)
        
        if args.interactive:
            client.chat_interactive(args.system)
        elif args.prompt:
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": args.prompt})
            
            if config.stream:
                print("Assistant: ", end="", flush=True)
                for chunk in client.get_completion(messages):
                    print(chunk, end="", flush=True)
                print()  # New line after streaming
            else:
                response = client.get_completion(messages)
                print(f"Assistant: {response.content}")
                
                if args.verbose:
                    print(f"\nMetadata:")
                    print(f"  Model: {response.model}")
                    print(f"  Usage: {response.usage}")
                    print(f"  Response time: {response.response_time:.2f}s")
                    print(f"  Finish reason: {response.finish_reason}")
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
