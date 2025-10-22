

from langchain.messages import AIMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash")


# Create an AI message manually (e.g., for conversation history)
ai_msg = AIMessage("I'd be happy to help you with that question!")

# Add to conversation history
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg,  # Insert as if it came from the model
    HumanMessage("Great! What's 2+2?")
]

response = model.invoke(messages)
print(response)