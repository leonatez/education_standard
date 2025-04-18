from dotenv import load_dotenv
import tempfile
import os
import torch
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
# from llama_cloud_services import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, get_response_synthesizer, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow, AgentOutput, ToolCall, ToolCallResult
from llama_index.core.workflow import (
    step,   
    Context,
)
from llama_index.core.prompts import PromptTemplate

import shutil
import traceback
import nest_asyncio
nest_asyncio.apply()

# === Load environment ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment or .env")

# === Fix torch bug if needed ===
torch.classes.__path__ = []
# llama_index.core.set_global_handler("simple")

# === FastAPI app setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.middleware("http")
# async def log_requests(request, call_next):
#     print("Incoming request headers:", dict(request.headers))
#     response = await call_next(request)
#     return response

# === LLM + Parser ===
llm = OpenAI(api_key=OPENAI_API_KEY)
# parser = LlamaParse(
#     result_type="markdown",
#     parse_mode="parse_document_with_llm",
#     auto_mode_trigger_on_table_in_page=True,
#     auto_mode_trigger_on_image_in_page=True,
#     continuous_mode=True,
#     spreadsheet_extract_sub_tables=True
# )
# ====USING MARKITDOWN====
from markitdown import MarkItDown
md = MarkItDown(enable_plugins=False) # Set to True to enable plugins


# === Tools warehouse ===
async def search_documents(query: str) -> str:
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    response = await query_engine.aquery(query)
    return str(response)

#Our record_notes tool will access the current state, add the notes to the state,
#  and then return a message indicating that the notes have been recorded.
async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic."""
    current_state = await ctx.get("state")
    if "research_notes" not in current_state:
        current_state["research_notes"] = {}
    current_state["research_notes"][notes_title] = notes
    await ctx.set("state", current_state)
    return "Notes recorded. Handoff to StandardExpertAgent"

#write_report and review_report will similarly be tools that access the state:
async def write_report(ctx: Context, report_content: str) -> str:
    """Useful for writing a report on a given topic."""
    current_state = await ctx.get("state")
    current_state["report_content"] = report_content
    await ctx.set("state", current_state)
    return "Report written."


async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback."""
    current_state = await ctx.get("state")
    current_state["review"] = review
    await ctx.set("state", current_state)
    return "Report reviewed."


# === Upload and store documents ===
@app.post("/uploadfiles/")
async def upload_files(files: List[UploadFile]):
    model = "text-embedding-3-large"
    Settings.embed_model = OpenAIEmbedding(model=model)
    print("Start working, embedding model:", model)
    try:
        file_paths = []

        for file in files:
            # Extract original file extension
            original_extension = os.path.splitext(file.filename)[1]
            
            # Create a temporary file with the original extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=original_extension) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp.flush()
                file_paths.append(tmp.name)
        print("file paths:", file_paths)
        # Parse using LlamaParse
        # documents = parser.load_data(file_paths)
        # Parse using Markitdown
        join_documents = [md.convert(file_path) for file_path in file_paths]
        documents_text = "\n".join([doc.text_content for doc in join_documents])
        print("Successfully parsed documents", documents_text)

        # Index and persist
        documents = [Document(text=doc.text_content) for doc in join_documents]
        index = VectorStoreIndex.from_documents(documents) #for llamaparse just feed documents is enough
        index.storage_context.persist(persist_dir="storage")

        # Optional: clean up files afterward
        for path in file_paths:
            os.remove(path)

        return {"message": "Inserted and stored files successfully"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


#=== Query with uploaded files + existing index ===
@app.post("/query/")
async def query_with_context(files: List[UploadFile]):
    try:
        # Step 1: Parse new uploaded files
        file_paths = []

        for file in files:
            # Extract original file extension
            original_extension = os.path.splitext(file.filename)[1]
            
            # Create a temporary file with the original extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=original_extension) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp.flush()
                file_paths.append(tmp.name)
        print("file paths:", file_paths)
        # Parse using LlamaParse
        # new_documents = parser.load_data(file_paths)
        # Parse using Markitdown
        join_new_documents = [md.convert(file_path) for file_path in file_paths]
        new_documents = "\n".join([doc.text_content for doc in join_new_documents])
        # new_documents = "Diện tích sàn xây dựng phục vụ đào tạo trên số người học chính quy quy đổi theo trình độ và lĩnh vực đào tạo của chúng tôi là 15 m2; 60% giảng viên toàn thời gian được bố trí chỗ làm việc riêng biệt."
        print("Successfully parsed documents: ", new_documents)

        

        # Step 4: Build the agents
        model = OpenAI(model="gpt-4o")
        record_agent = FunctionAgent(
            name="RecordAgent",
            description="Hữu ích trong việc nhận báo cáo tình hình thực tế triển khai của trường đại học rồi tổng hợp thành ghi chú. After summarizing the report, hand off to StandardExpertAgent.",
            tools=[record_notes],
            llm=model,
            system_prompt=(
                "Bạn là cán bộ tiếp nhận báo cáo từ các trường đại học về "
                "tình hình triển khai thực tế của họ. "
                "Công việc của bạn là đọc báo cáo rồi tổng hợp các thông tin được đề cập trong đó thành "
                "đầy đủ các hạng mục để hỗ trợ các cán bộ khác đánh giá tiêu chuẩn sau này. "
                "Bạn phải ghi rõ là bạn đã nhận báo cáo gì rồi chỉ cần tổng hợp các hạng mục và không cần phải đánh giá. "
                "Lưu ý không được tự suy diễn mà chỉ tổng hợp từ thông tin được đề cập trong báo cáo. "
                "Kết quả công việc của bạn là tách nhỏ các hạng mục và đặt câu hỏi về quy định của hạng mục đó "
                "Sau khi tổng hợp xong hãy bàn giao cho StandardExpertAgent"
                "If you have finished summarizing, respond with 'handoff to StandardExpertAgent'."
            ),
            can_handoff_to=["StandardExpertAgent"]
        )

        standard_expert = FunctionAgent(
            name="StandardExpertAgent",
            description="Hữu ích trong việc tìm kiếm quy định liên quan tới các hạng mục được nhắc đến trong thông tư ban hành Chuẩn cơ sở giáo dục đại học",
            tools=[search_documents],
            llm=model,
            system_prompt=(
                "Bạn là chuyên gia về thông tư ban hành Chuẩn cơ sở giáo dục đại học "
                "Công việc của bạn là tìm và trích xuất quy định về thông tư ban hành Chuẩn cơ sở giáo dục đại học liên quan tới các hạng mục được nhắc đến trong tổng hợp. "
                "Lưu ý không được tự suy diễn mà phải dùng tool search_documents và dựa theo kiến thức từ đó để trích xuất. "
                "Bạn chỉ trích xuất quy định có liên quan tới các hạng mục được nhắc đến trong tổng hợp và không được đánh giá gì về thực tế triển khai ở trường đại học. "
                "Vì vậy, trong kết quả trả ra hãy dẫn nguồn rõ ràng là từ điều mấy, mục nào, hoặc theo tiêu chí số mấy, thông tư nào."
                "Ví dụ: CÁC QUY ĐỊNH LIÊN QUAN ĐẾN THỰC TẾ TRIỂN KHAI:"
                "Hạng mục 1: Theo tiêu chí 2.1, thông tư 25/2015/TT-BYT, tỷ lệ người học trên giảng viên tối đa 40%"
                "Hạng mục 2: Theo tiêu chí 2.2, thông tư 25/2015/TT-BYT, tỷ lệ giảng viên cơ hữu trong độ tuổi lao động tối thiểu 70%"
                "Khi trích xuất xong, hãy bàn giao kết quả của bạn cho QCOfficerAgent. "
                "If you have finished searching, respond with 'handoff to QCOfficerAgent'."
            ),
            can_handoff_to=["QCOfficerAgent"],
        )

        qc_officer = FunctionAgent(
            name="QCOfficerAgent",
            description="Hữu ích trong việc báo cáo đánh giá tình hình thực tế triển khai của trường đại học",
            tools=[write_report,review_report],
            llm=model,
            system_prompt=(
                "Bạn là cán bộ trưởng phòng kiểm định chất lượng của Bộ Giáo Dục Việt Nam. "
                "Công việc của bạn là báo cáo đánh giá theo dạng markdown về tình hình thực tế triển khai của trường đại học so sánh, dựa theo thông tư ban hành Chuẩn cơ sở giáo dục đại học. "
                "Bạn sẽ chỉ đạo RecordAgent tổng hợp báo cáo tình hình thực tế và StandardExpertAgent trích xuất quy định. "
                "Bạn chỉ được đánh giá các hạng mục mà RecordAgent đã tổng hợp, dựa theo thông tư ban hành Chuẩn cơ sở giáo dục đại học mà StandardExpertAgent đã trích xuất. "
                "Báo cáo của bạn phải đi tuần tự từng hạng mục, gồm thực tế đã triển khai gì (Thực tế triển khai), "
                "quy định của thông tư ban hành Chuẩn cơ sở giáo dục đại học về hạng mục đó ra sao (Quy định) và "
                "thực tế triển khai đã đạt yêu cầu không (Đánh giá). "
                "Ví dụ: ĐÁNH GIÁ CHẤT LƯỢNG TRIỂN KHAI CỦA TRƯỜNG ĐẠI HỌC A:"
                "Hạng mục 1: Tỷ lệ người học trên giảng viên là 50."
                "Quy định: Tỷ lệ người học trên giảng viên là ≤ 40:1."
                "Đánh giá: Không đạt, do tỷ lệ người học trên giảng viên là 50, vượt quá yêu cầu 40:1."
                "Hạng mục 2: Tỷ lệ giảng viên cơ hữu trong độ tuổi lao động là 15%."
                "Quy định: Tỷ lệ giảng viên cơ hữu trong độ tuổi lao động là ≥ 70%."
                "Đánh giá: Không đạt, do tỷ lệ giảng viên cơ hữu trong độ tuổi lao động thấp hơn quy định 70%."
                "Nếu bạn thấy không đủ thông tin hoặc không rõ, hãy yêu cầu RecordAgent hoặc StandardExpertAgent cung cấp thêm thông tin."
            ),
            can_handoff_to=["RecordAgent", "StandardExpertAgent"],
        )

        # Step 5: Run the agent with new documents as context
        
        agent_workflow = AgentWorkflow(
            agents=[record_agent, standard_expert, qc_officer],
            root_agent=record_agent.name,
            initial_state={
                "research_notes": {},
                "report_content": "Not written yet.",
                "review": "Review required.",
                "original_user_request": f"""
                    Đây là báo cáo về tình hình thực tế triển khai của trường đại học,
                    hãy tổng hợp, phân tích và đánh giá xem từng hạng mục thực tế triển khai này đã đạt yêu cầu theo
                    thông tư ban hành Chuẩn cơ sở giáo dục đại học chưa: {new_documents}
                    Kết quả phải là báo cáo hoàn chỉnh, bao gồm:
                    - Thực tế triển khai
                    - Quy định của thông tư ban hành Chuẩn cơ sở giáo dục đại học
                    - Đánh giá
                    """
            },
        )

        handler = agent_workflow.run(
            user_msg=f"""
            Đây là báo cáo về tình hình thực tế triển khai của trường đại học,
            hãy tổng hợp, phân tích và đánh giá xem từng hạng mục thực tế triển khai này đã đạt yêu cầu theo
            thông tư ban hành Chuẩn cơ sở giáo dục đại học chưa: {new_documents}
            Kết quả phải là báo cáo hoàn chỉnh dạng markdown, bao gồm:
            - Thực tế triển khai
            - Quy định của thông tư ban hành Chuẩn cơ sở giáo dục đại học
            - Đánh giá
            Phải đảm bảo tất cả agent đều hoạt động trong workflow.
            """
        )

        current_agent = None
        current_tool_calls = ""
        output_log = []
        async for event in handler.stream_events():
            if (
                hasattr(event, "current_agent_name")
                and event.current_agent_name != current_agent
            ):
                current_agent = event.current_agent_name
                print(f"\n{'='*50}")
                print(f"🤖 Agent: {current_agent}")
                print(f"{'='*50}\n")
                output_log.append(f"\n{'='*50}")
                output_log.append(f"🤖 Agent: {current_agent}")
                output_log.append(f"{'='*50}\n")
            elif isinstance(event, AgentOutput):
                if event.response.content:
                    print("📤 Output:", event.response.content)
                    output_log.append(f"📤 Output: {event.response.content}")
                if event.tool_calls:
                    print(
                        "🛠️  Planning to use tools:",
                        [call.tool_name for call in event.tool_calls],
                    )
                    output_log.append(f"🛠️  Planning to use tools: {', '.join([call.tool_name for call in event.tool_calls])}")
            elif isinstance(event, ToolCallResult):
                print(f"🔧 Tool Result ({event.tool_name}):")
                print(f"  Arguments: {event.tool_kwargs}")
                print(f"  Output: {event.tool_output}")
                output_log.append(f"🔧 Tool Result ({event.tool_name}):")
                output_log.append(f"  Arguments: {event.tool_kwargs}")
                output_log.append(f"  Output: {event.tool_output}")
            elif isinstance(event, ToolCall):
                print(f"🔨 Calling Tool: {event.tool_name}")
                print(f"  With arguments: {event.tool_kwargs}")
                output_log.append(f"🔨 Calling Tool: {event.tool_name}")
                output_log.append(f"  With arguments: {event.tool_kwargs}")
            # print(handler.state)
        return {"output_log": output_log}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

# === Run App ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
