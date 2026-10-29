import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# RAG 相关依赖
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="我的智能 PDF 问答助手 API")

# ================= 1. 初始化 RAG 引擎 =================
print("正在初始化 RAG 引擎，请稍候...")
loader = PyPDFLoader("rules.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="text-embedding-v3",
    check_embedding_ctx_length=False,
    chunk_size=10
)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
)

template = """请仅根据以下上下文来回答问题。如果不知道，请直接说不知道，不要瞎编。
上下文：
{context}

问题：
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("RAG 引擎初始化完毕！")
# ====================================================

# 定义接口的请求体
class QuestionRequest(BaseModel):
    question: str

# 新增接口：基于 PDF 的问答
@app.post("/ask_pdf")
async def ask_pdf(req: QuestionRequest):
    try:
        # 调用之前组装好的 RAG 链路
        answer = rag_chain.invoke(req.question)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}