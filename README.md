# deepseek-rag-api

# 基于 FastAPI 与 LangChain 的 RAG 智能问答 API

## 项目简介
这是一个基于大模型的本地文档智能问答后端项目。用户可以通过 RESTful 接口上传问题，系统会自动从本地 PDF 文档中检索相关内容，并调用大模型生成基于文档上下文的准确回答，有效解决大模型“幻觉”问题。

## 技术栈
*   **后端框架**：FastAPI
*   **AI 框架**：LangChain
*   **向量数据库**：Chroma
*   **大模型**：阿里云百炼 (OpenAI 兼容接口)
*   **文档解析**：pypdf

## 核心功能
1. **API 封装**：提供 `/ask_pdf` 接口，接收 JSON 请求并返回智能回答。
2. **RAG 检索增强**：自动解析本地 PDF，切片并向量化存入 Chroma，实现 Top-3 语义检索。
3. **安全隔离**：使用 `.env` 管理 API Key，结合虚拟环境隔离依赖。

## 本地运行说明
1. 克隆项目并进入目录
2. 创建虚拟环境并激活：`python -m venv .venv`
3. 安装依赖：`pip install -r requirements.txt`
4. 配置 `.env` 文件中的 API Key（参考 `.env.example`）
5. 启动服务：`uvicorn app_new:app --reload`
6. 访问 `http://127.0.0.1:8000/docs` 进行接口测试
