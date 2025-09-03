# Simple usage
from openai_client import get_completion
response = get_completion("Explain quantum computing in simple terms")
print(response.content)

# Advanced usage
from openai_client import ChatClient, ChatConfig
config = ChatConfig(model="gpt-4o", temperature=0.3, stream=True)
client = ChatClient(config=config)
