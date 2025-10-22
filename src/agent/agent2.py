#langchain version>1.0
#简单的agent示例
from langchain.agents import create_agent

from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
result=agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in shanghai"}]}
)

print(result)