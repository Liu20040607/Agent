import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import Tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

@st.cache_resource
def get_embedding_model():
    print("初始化 Embedding 模型")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    print("Embedding 模型初始化完畢")
    return embeddings_model

@st.cache_resource
def get_default_retriever():
    print("正在載入預設的知識庫")
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=get_embedding_model()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("載入完畢")
    return retriever

def get_retriever_for_pdf(uploaded_file):
    """
    處理上傳的 PDF 檔案，建立並回傳一個臨時的 RAG Retriever。
    如果處理失敗，回傳 None。
    """
    print(f"正在處理上傳的檔案: {uploaded_file.name} ")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = get_embedding_model()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        os.remove(tmp_file_path)
        return retriever
    
    except Exception as e:
        print(f"--- [read_paper] 處理檔案時發生錯誤：{e} ---")
        st.error(f"處理檔案 '{uploaded_file.name}' 時發生錯誤：{e}")
        return None 


def create_rag_chain(retriever, llm):
    print("正在建立 RAG 鏈...")
    rag_template = """
    Answer the user's question based ONLY on the following context.
    You MUST cite the source for your answer using the [第 X 頁] format.
    If the context doesn't contain the answer, say "根據提供的上下文，我找不到答案".

    Context:
    {context}
    
    Question:
    {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    def format_docs_with_sources(docs):
        format_context =""
        for i,doc in enumerate(docs):
            pagenum = doc.metadata.get("page", 0) + 1
            sourceid = f"[第 {pagenum} 頁]"
            format_context += f"{sourceid}\n{doc.page_content}\n\n---\n\n"
        return format_context

    # RAG 定義
    rag_chain = (
        {"context": retriever | format_docs_with_sources, 
         "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("RAG 建立完畢")
    return rag_chain

def create_paper_qa_tool(rag_chain, paper_name):

    print(f"正在建立 PaperQA 工具 (for {paper_name})...")
    paper_qa_tool = Tool(
        name="PaperQA",
        description=f"Use this tool ONLY when the user asks questions " \
                    f"about the academic paper '{paper_name}'.",
        func=rag_chain.invoke # 呼叫此工具時，執行 rag_chain
    )
    print("PaperQA 工具建立完畢")
    return paper_qa_tool