import getpass
import os
from dotenv import load_dotenv
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers  import StrOutputParser

system_template = "Translate the following from English into {language}"
    
try:
    # load environment variables from .env file (requires `python-dotenv`)
   

    load_dotenv()
except ImportError:
    pass
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

prompt_template = ChatPromptTemplate.from_template(
    "给我讲个关于的{topic}笑话"
)

parser=StrOutputParser()

chain=prompt_template | model | parser




async def async_stream():
    # async for chunk in chain.astream({"topic":"猫"}):
    
    model1 = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    async for chunk in model1.astream("讲个笑话"):
        print(chunk.content, end="|", flush=True)

async def async_stream2():
    
    model2 = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    async for chunk in model2.astream("天空是什么颜色"):
        print(chunk.content, end="|", flush=True)


async def main():  #同步按循序调用
    await async_stream()
    await async_stream2()

async def main2():  #并发运行
    await asyncio.gather(async_stream(), async_stream2())

asyncio.run(main2())


# async def async_run():
#     res=await chain.arun({"topic":"猫"})
#     print(res)
