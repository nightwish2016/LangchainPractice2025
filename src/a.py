import getpass
import os
from dotenv import load_dotenv

try:
    # load environment variables from .env file (requires `python-dotenv`)
   

    load_dotenv()
except ImportError:
    pass
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi，how are you! nice to meet you."),
]

# res=model.invoke(messages)
# print(res)

for token in model.stream(messages):
    print(token.content, end="|")

# model.invoke("Hello")

# model.invoke([{"role": "user", "content": "Hello"}])

# model.invoke([HumanMessage("Hello")])