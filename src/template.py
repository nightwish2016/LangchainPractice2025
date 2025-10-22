from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

try:
    # load environment variables from .env file (requires `python-dotenv`)
   

    load_dotenv()
except ImportError:
    pass

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)


prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

# prompt
from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
response = model.invoke(prompt)
print(response.content)