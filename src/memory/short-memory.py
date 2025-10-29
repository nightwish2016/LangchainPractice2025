from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# 1️⃣ 初始化模型
llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0)

@tool
def get_user_info(name: str) -> str:
    """Get info about the user"""
    return f"User's name is {name}"


agent = create_agent(
    llm,
    [get_user_info],
    checkpointer=InMemorySaver(),  
)

res=agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},  
)
print(res.content)


res2=agent.invoke(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    {"configurable": {"thread_id": "1"}},  
)
print(res2)