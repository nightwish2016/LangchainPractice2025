#动态模型

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

from dotenv import load_dotenv

load_dotenv()


basic_model = init_chat_model(model="google_genai:gemini-2.5-flash")
advanced_model = init_chat_model(model="google_genai:gemini-2.5-flash-lite")


# model = init_chat_model(
#     "google_genai:gemini-2.5-flash",
#     temperature=0
# )




@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 1:
        # Use advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)



@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# agent = create_agent(model, tools=[get_weather])



agent = create_agent(
    model=basic_model,  # Default model
    tools=[get_weather],
    middleware=[dynamic_model_selection]
)

messages = [{"role": "user", "content": "你好"}]
for i in range(12):
    messages.append({"role": "assistant", "content": f"第{i}轮回复"})
    messages.append({"role": "user", "content": f"第{i}轮提问"})

result=agent.invoke(
    {"messages": messages}
)

print(result)