# 論文 AI 助理

**本專案使用 LangChain 框架，結合了 RAG 與 Agent ，打造了一個能理解特定論文 (預設為 GaitSet，並支援動態上傳 PDF) 且能自主上網查詢即時資訊的 AI 助理。**

---

##  Demo 展示

<!-- ```markdown -->
[Demo 展示](https://www.youtube.com/watch?v=LzKNy8TuxbQ)

---
## 核心功能
本專案的 AI Agent 整合了多項技術，具備以下核心能力：

    PaperQA 工具:

    預設知識庫: 內建 GaitSet 論文知識庫 (./chroma_db)。

    動態上傳: 使用者可透過 Streamlit 側邊欄上傳任意 PDF，系統會即時建立臨時 RAG 知識庫。

    引用來源: AI 回答論文相關問題時，會自動引用 PDF 的頁碼 ([第 X 頁])，確保答案的可追溯性與嚴謹性。

    技術細節: PyPDFLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings (all-MiniLM-L6-v2), ChromaDB 。

    WebSearch 工具:

    即時搜尋: 整合 Tavily AI，能回答即時資訊 (新聞、天氣、軟體版本) 或論文中未提及的內容。

    ReAct 框架:

    1.Agent 根據問題語意，自主判斷應使用 PaperQA (論文問題) 或 WebSearch (其他問題)。

    2.當 PaperQA 在論文中找不到答案時，Agent 會自動觸發 WebSearch，嘗試從網路尋找答案。

    3.Agent 能記住先前的對話內容，支援多輪追問，提供更連貫的互動。

    4.使用 Streamlit 打造，介面簡潔直觀，包含聊天記錄顯示、檔案上傳、載入狀態提示等。

---
## 使用技術
    語言： Python

    AI 框架： LangChain (Agents, LCEL, RAG, Memory)

    LLM (大腦)： Google Gemini 2.5 Flash

    Embedding (向量化)： HuggingFace all-MiniLM-L6-v2 (本機)

    向量資料庫： ChromaDB (本機)

    Web UI 框架： Streamlit

    工具 API： Tavily AI