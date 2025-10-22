#tool的调用
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    temperature=0
)

agent = create_agent(model, tools=[search, get_weather])
result=agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in shanghai"}]}
)
print(result)