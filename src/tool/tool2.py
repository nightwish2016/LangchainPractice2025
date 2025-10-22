# error handling
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

load_dotenv()

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # if not location:
    #     raise ValueError("Location cannot be empty")
    # return f"Weather in {location}: Sunny, 72°F"
    raise RuntimeError("Fake search failure")



@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
    

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    temperature=0
)

agent = create_agent(model, tools=[search, get_weather], middleware=[handle_tool_errors])
result=agent.invoke(
    {"messages": [{"role": "user", "content": "NEW YORK wether"}]}
)
print(result)