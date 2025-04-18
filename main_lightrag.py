import os
import tempfile
import torch
import numpy as np
# import google.generativeai as genai
# from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from openai import OpenAI

from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition
from fastapi import FastAPI, File, UploadFile
from typing import List

torch.classes.__path__ = []
app = FastAPI()

# =======================SETTING UP===============================

load_dotenv()

from fastapi.middleware.cors import CORSMiddleware


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # This must match EXACTLY with frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # or ["POST"]
    allow_headers=["*"],  # You can restrict if needed
)

@app.middleware("http")
async def log_requests(request, call_next):
    print("Incoming request headers:", dict(request.headers))
    response = await call_next(request)
    return response


# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

WORKING_DIR = "./data"

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in environment variable or .env file.")

# Initialize Gemini model directly
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel("gemini-2.0-flash-lite")

#=============================LIGHTRAG SETUP==============================
async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    print("\n\n=== llm_model_func called ===")
    print("System prompt:", system_prompt)
    print("Prompt:", prompt)
    print("============================\n\n")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content

    # return await openai_complete_if_cache(
    #     "ChatGPT-4o",
    #     prompt,
    #     system_prompt=system_prompt,
    #     history_messages=history_messages,
    #     api_key=OPENAI_API_KEY,
    #     base_url="https://api.openai.com/v1/chat/completions",
    #     **kwargs
    # )

async def embedding_func(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([item.embedding for item in response.data])  # ✅ batch of embeddings


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=20000,
        func=embedding_func,
    ),
)

#==================FASTAPI===============================

@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile]):
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        elements = partition(tmp_path)
        text = "\n".join([el.text for el in elements])
        print("======> Unstructured Text:",text)
        await rag.ainsert(text)
        print(f"Received file: {file.filename}")

    return "Inserted file successfully"  # Return filenames for confirmation

@app.post("/query/")
async def query(files: List[UploadFile]):
    query_param = QueryParam(
        mode="hybrid", 
        top_k=20, 
        response_type="Multiple Paragraphs",
        max_token_for_text_unit=50000,
        max_token_for_global_context=40000,
        max_token_for_local_context=20000
    )
    
    all_text = ""

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        elements = partition(tmp_path)
        text = "\n".join([el.text for el in elements])
        all_text += f"\n{text}"  # Append text from each file
        print(f"Parsed file: {file.filename}, Length: {len(text)}")

    print("======> Combined Unstructured Text:", all_text[:1000], "...")  # preview only first 1000 chars

    response = await rag.aquery(
        f"Đây là thông tin về tình hình thực tế triển khai đảm bảo chỉ tiêu Chuẩn cơ sở giáo dục đại học của một trường đại học, hãy phân tích và đánh giá theo các tiêu chí trong tài liệu xem việc triển khai này đã đảm bảo chất lượng chưa: {all_text}",
        param=query_param,
        system_prompt="Bạn là cán bộ kiểm định chất lượng của Bộ Giáo Dục Việt Nam. Nhiệm vụ của bạn là phân tích thông tin về tình hình thực tế triển khai đảm bảo chỉ tiêu Chuẩn cơ sở giáo dục đại học của một trường đại học và báo cáo chi tiết các chỉ tiêu nào đạt và không đạt cùng lý do chi tiết vì sao không đạt."
    )

    print("======> Gemini RAG Response:", response)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)