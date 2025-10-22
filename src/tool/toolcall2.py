from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from langchain.messages import AIMessage, SystemMessage, HumanMessage,ToolMessage
load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash")


# After a model makes a tool call
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# Execute tool and create result message
weather_result = "Sunny, 72°F"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # Must match the call ID
)

# Continue conversation
messages = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,  # Model's tool call
    tool_message,  # Tool execution result
]
response = model.invoke(messages)  # Model processes the result
print(response.content)  # Model's final response