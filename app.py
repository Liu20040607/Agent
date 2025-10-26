import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

import read_paper 

@st.cache_resource
def get_llm_and_websearch():
    print("正在初始化 LLM 和 WebSearch")
    load_dotenv()
    google_key = os.getenv("GOOGLE_GENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if google_key is None or tavily_key is None:
        raise ValueError("找不到 GOOGLE_GENAI_API_KEY 或 TAVILY_API_KEY。")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_key
    )
    web_search_tool = TavilySearchResults(
        name="WebSearch",
        description="用於搜尋網路上的即時資訊、新聞、軟體和" \
                    "「目前載入的論文」中未提及的資訊。",
        tavily_api_key=tavily_key
    )
    print("LLM 和 WebSearch 初始化完畢")
    return llm, web_search_tool


agent_prompt_template_text = """
你是一個論文AI 助理，你可以使用以下工具：
{tools}

你必須使用以下格式來回應：
Thought: [你的思考過程，你計畫做什麼]
Action: [你要使用的工具名稱，必須是 {tool_names} 之一]
Action Input: [工具的輸入]
Observation: [工具的回傳結果]
...
Thought: [我現在有足夠的資訊來回答了]
Final Answer: [給使用者的最終答案]
---
#特殊要求:
1.  如果問題是關於「目前載入的學術論文」... 你必須優先使用 `PaperQA` 工具。如果問題是關於 "即時資訊", "新聞", "天氣" 等等，你必須優先使用 `WebSearch` 工具。
2. 只有在你優先使用了 `PaperQA` 之後，並且 `PaperQA` 回傳了「`根據提供的上下文，我找不到答案`」這句話，你才可以接著使用 `WebSearch` 嘗試尋找同一個問題的答案。
3. 當你使用 `PaperQA` 工具時，請在回答中引用相關的頁碼。例如: 「根據第 3 頁，...」 
4. 當你使用 `WebSearch` 工具時，請在回答中引用相關的來源。例如: 「根據 [來源名稱]，...」
5. 請用繁體中文回答所有問題。
---
# 歷史紀錄:
{chat_history}
# 目前對話:
Question: {input}
Thought: {agent_scratchpad}

"""
agent_prompt = ChatPromptTemplate.from_template(agent_prompt_template_text)

st.set_page_config(page_title="論文 AI 助理")
st.caption(f"支援 PDF 上傳 ")

with st.sidebar:
    st.title("知識庫設定")
    st.info("可以選擇使用預設的 GaitSet 論文，或上傳你自己的 PDF。")
    
    uploaded_file = st.file_uploader("上傳你的 PDF 論文", type="pdf")
    
    if "session_retriever" not in st.session_state:
        st.session_state.session_retriever = read_paper.get_default_retriever()
        st.session_state.paper_name = "預設論文 (GaitSet)"

    if uploaded_file is not None:
        if st.session_state.paper_name != uploaded_file.name:
            with st.spinner(f"正在處理 {uploaded_file.name}..."):
                new_retriever = read_paper.get_retriever_for_pdf(uploaded_file)
                
                if new_retriever is not None: # 檢查是否處理成功
                    st.session_state.session_retriever = new_retriever
                    st.session_state.paper_name = uploaded_file.name
                    st.success(f"已成功載入 {uploaded_file.name}！")
                    st.session_state.messages = [
                         {"role": "assistant", "content": f"你好！我已載入「{st.session_state.paper_name}」。你可以開始問我問題了。"}
                    ]
                    st.rerun()
                else:
                    st.session_state.session_retriever = read_paper.get_default_retriever()
                    st.session_state.paper_name = "預設論文 (GaitSet)"
    st.info(f"目前知識庫: {st.session_state.paper_name}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"你好！我已載入「{st.session_state.paper_name}」知識庫。你可以開始問我問題了。"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


llm, web_search_tool = get_llm_and_websearch()


if prompt := st.chat_input(""):
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    current_retriever = st.session_state.session_retriever
    rag_chain = read_paper.create_rag_chain(current_retriever, llm)
    paper_qa_tool = read_paper.create_paper_qa_tool(rag_chain, st.session_state.paper_name)
    tools = [web_search_tool, paper_qa_tool]

    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考中... "):
            try:
                chat_history = []
                for msg in st.session_state.messages: 
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                agent = create_react_agent(llm=llm, tools=tools, prompt=agent_prompt)
                agent_executor = AgentExecutor(
                    agent=agent, tools=tools, verbose=True, 
                    handle_parsing_errors=True, max_iterations=10 
                )
                
                def stream_agent_response():
                    response_stream = agent_executor.stream({
                        "input": prompt, "chat_history": chat_history 
                    })
                    final_output = ""
                    for chunk in response_stream:
                        if "output" in chunk:
                            token = chunk["output"]
                            final_output += token
                            yield token
                    st.session_state.last_response = final_output 

                response_content = st.write_stream(stream_agent_response())
                

                if response_content is None and hasattr(st.session_state, 'last_response'):
                     response_content = st.session_state.last_response
                elif response_content is None:
                     response_content = "[串流輸出錯誤或為空]" 

            except Exception as e:
                if "ResourceExhausted" in str(e) or "429" in str(e):
                    response_content = "API 額度已耗盡 "
                else:
                    response_content = f"Agent 執行錯誤：\n\n{str(e)}"
                st.markdown(response_content) # 錯誤直接印出

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append(
        {"role": "assistant", "content": response_content}
    )
    