from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

# model = init_chat_model("openai:gpt-4.1-mini")

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    temperature=0
)

tool = {"type": "web_search"}
model_with_tools = model.bind_tools([tool])

response = model_with_tools.invoke("What was a positive news story from today?")
print(response.content_blocks)