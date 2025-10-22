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

# chain=prompt_template | model | parser




async def async_stream():
    events=[]
    async for chunk in model.astream_events("hello"):
        events.append(chunk)
        # print(chunk, end="|", flush=True)
    print(events)



asyncio.run(async_stream())

# async def async_run():
#     res=await chain.arun({"topic":"猫"})
#     print(res)
