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

# vector_store = InMemoryVectorStore(embeddings) #saave in memory

from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
) #save vector in Choma database

ids = vector_store.add_documents(documents=all_splits)

# results = vector_store.similarity_search(
#     "我擅长什么"
# )
# print("_________________")
# print(results[0])



# async def main():
#     results = await vector_store.asimilarity_search("我擅长什么?")
#     print(results[0].page_content)



# asyncio.run(main())



results = vector_store.similarity_search_with_score("我擅长什么")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)





# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 1},
# )

# res=retriever.batch(
#     [
#         "good这个单词在里面吗"
       
#     ],
# )
# print("===============")
# print(res[0])
# print(res[1])