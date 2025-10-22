from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
load_dotenv()

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

# result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
print(result["structured_response"])