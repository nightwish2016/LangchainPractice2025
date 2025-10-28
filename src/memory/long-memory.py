from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, StateGraph, START, END

load_dotenv()

llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0)


def run_demo() -> None:
    with SqliteSaver.from_conn_string("memory.sqlite") as checkpointer:
        graph = StateGraph(MessagesState)

        def call_model(state: MessagesState):
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        graph.add_node("model", call_model)
        graph.add_edge(START, "model")
        graph.add_edge("model", END)

        agent = graph.compile(checkpointer=checkpointer)

        result1 = agent.invoke(
            {"messages": [HumanMessage(content="Hi, my name is Kevin. I live in Shanghai.")]},
            {"configurable": {"thread_id": "user-001"}},
        )
        print(result1["messages"][-1].content)

        result2 = agent.invoke(
            {"messages": [HumanMessage(content="Do you remember where I live and who I am?")]},
            {"configurable": {"thread_id": "user-001"}},
        )
        print(result2["messages"][-1].content)


if __name__ == "__main__":
    run_demo()