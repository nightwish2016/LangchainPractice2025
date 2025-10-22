from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.structured_output import ProviderStrategy
from dotenv import load_dotenv
import json

load_dotenv()

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

# ✅ 使用 ProviderStrategy，告诉模型用 provider 原生的 structured output 机制
agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[search_tool],
    response_format=ProviderStrategy(ContactInfo)
)

# 模型生成结构化结果
result = agent.invoke("Extract contact info from: John Doe, john@example.com, (555) 123-4567")

# result 是 Pydantic 对象
print(result)
print("\nJSON format:")
print(json.dumps(result.model_dump(), indent=2))
