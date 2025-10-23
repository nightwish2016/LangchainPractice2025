# from langchain_core.documents import Document

# documents = [
#     Document(
#         page_content="Dogs are great companions, known for their loyalty and friendliness.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
#     Document(
#         page_content="Cats are independent pets that often enjoy their own space.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
# ]

#导入文档
import asyncio
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader

file_path = "./resume.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
# print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)


#分片
from langchain_text_splitters   import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))


#向量化
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])


#向量数据库存储

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

# results = vector_store.similarity_search(
#     "我擅长什么", k=2
# )
# print("_________________******")
# print(results[0])



# async def main():
#     results = await vector_store.asimilarity_search("我擅长什么?")
#     print(results[0].page_content)



# asyncio.run(main())



# results = vector_store.similarity_search_with_score("我擅长什么")
# doc, score = results[0]
# print(f"Score: {score}\n")
# print(doc)

from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # results = vector_store.similarity_search("我擅长什么")
    # for r in results:
    #     print("___________")
    #     print(r.page_content[:200])
    return serialized, retrieved_docs



from langchain.agents import create_agent


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "你是一个专业的简历查询助手，你的唯一目标是回答用户的简历相关问题。你有一个名为 'retrieve_context' 的工具，它连接到用户的简历数据。对于任何关于用户的技能、经验或任何涉及 '我' 的个人化查询，你**必须**使用 'retrieve_context' 工具来检索信息，并基于检索到的上下文生成详细的回答。"
    # "Use the tool to help answer user queries."
)


from langchain.chat_models import init_chat_model
model = init_chat_model("google_genai:gemini-2.5-flash")


agent = create_agent(model, tools,system_prompt=prompt)



query = (
    "如果从毕业时间开始计算，我工作几年了"
  
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()



