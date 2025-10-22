from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # 实际实现应该调用搜索API
    return f"Search results for: {query}"

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    temperature=0
)

# 正确的工具绑定方式
model_with_tools = model.bind_tools([web_search])

try:
    response = model_with_tools.invoke("What was a positive news story from today?")
    print(response.content_blocks)
except Exception as e:
    print(f"Error occurred: {e}")