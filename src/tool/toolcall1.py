from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash")

# model = init_chat_model("openai:gpt-5-nano")

def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return "It's sunny and 75°F in " + location + "."


model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in Paris?")

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")