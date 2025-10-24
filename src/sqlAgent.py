import requests, pathlib
from dotenv import load_dotenv
load_dotenv()
#Configure the database

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"{local_path} already exists, skipping download.")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"File downloaded and saved as {local_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")



from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')


# Add tools for database interactions
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain.chat_models import init_chat_model
model = init_chat_model("google_genai:gemini-2.5-flash")
from langchain_core.tools import tool

toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")

# Use create_agent
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

# from langchain.agents import create_agent


# agent = create_agent(
#     model,
#     tools,
#     system_prompt=system_prompt,
# )

# #run agent
# question = "Which genre on average has the longest tracks?"

# for step in agent.stream(
#     {"messages": [{"role": "user", "content": question}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()




from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 


agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[ 
        HumanInTheLoopMiddleware( 
            interrupt_on={"sql_db_query": True}, 
            description_prefix="Tool execution pending approval", 
        ), 
    ], 
    checkpointer=InMemorySaver(), 
)


question = "Which genre on average has the longest tracks?"
config = {"configurable": {"thread_id": "1"}} 
from langgraph.types import Command 
for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    # Command(resume=[{"type": "accept"}]), 
    config, 
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    elif "__interrupt__" in step: 
        print("INTERRUPTED:") 
        interrupt = step["__interrupt__"][0] 
        # 兼容不同类型的中断值
        if isinstance(interrupt.value, list):
            for request in interrupt.value:
                print(request if isinstance(request, str) else request.get("description", request))
        else:
            print(interrupt.value)
    else:
        pass



from langgraph.types import Command 

for step in agent.stream(
    # Command(resume=[{"type": "accept"}]), 
    Command(resume={"decisions": [{"type": "approve"}]}),
    config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    elif "__interrupt__" in step:
        print("INTERRUPTED:") 
        interrupt = step["__interrupt__"][0] 
        # 兼容不同类型的中断值
        if isinstance(interrupt.value, list):
            for request in interrupt.value:
                print(request if isinstance(request, str) else request.get("description", request))
        else:
            print(interrupt.value)
    else:
        pass