from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 1️⃣ 初始化模型
llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0)

# 2️⃣ 定义结构化输出 Schema
class MovieInfo(BaseModel):
    title: str = Field(description="The title of the movie")
    genre: str = Field(description="The genre of the movie")
    rating: float = Field(description="The rating of the movie out of 10")

# 3️⃣ 使用结构化输出
# structured_llm = llm.with_structured_output(MovieInfo)

# result = structured_llm.invoke("Tell me about the movie supper man")
# print(result)

#json输出
structured_llm = llm.with_structured_output(MovieInfo, include_raw=True)
result = structured_llm.invoke("Tell me about Titanic")

print("Raw JSON output:")
print(result["raw"])     # ✅ 正确访问 JSON 字符串

print("\nParsed object:")
print(result["parsed"])  # ✅ Pydantic 对象