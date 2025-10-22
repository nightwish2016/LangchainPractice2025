# langchain version<1.0，简单的agent示例
# 导入所需的类和函数
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. 定义工具函数
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 2. 封装工具
tools = [
    Tool(
        name="get_weather",
        func=get_weather,
        description="Get weather for a given city"
    )
]

# 3. 初始化 LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 4. 定义 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the tools provided when necessary."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 5. 创建 Agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# 6. 创建 Agent 执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 7. 运行
result = agent_executor.invoke({"input": "what is the weather in Shanghai?"})
print(result)
