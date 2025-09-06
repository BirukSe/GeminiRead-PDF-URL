import streamlit as st
from assistant import get_groq_assistant
import tempfile
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.web_base import WebBaseLoader

st.set_page_config(
    page_title="Groq RAG",
    page_icon=":orange_heart"
)
st.title("RAG with Gemini")
st.markdown(
"##### :orange_heart: built using [langchain](https://github.com/langchain)"
)
def restart_assistant():
    st.session_state["rag_assistant"]=None
    st.session_state["rag_assistant_run_id"]=None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"]+=1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"]+=1
    st.rerun()
def main()->None:
    llm_model=st.sidebar.selectbox("Select LLM",options=["gemini-1.5-flash","llama3-8b-8192","mixtral-8x7b-32768"])
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"]=llm_model
    elif st.session_state["llm_model"]!=llm_model:
        st.session_state["llm_model"]=llm_model
        restart_assistant()
    embeddings_model=st.sidebar.selectbox(
        "Select Embeddings",
        options=["all-MiniLM-L6-v2","text-embedding-3-small"],
        help="When you change the embeddings model, the documents will need to be added again"
    )
    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"]=embeddings_model
    elif st.session_state["embeddings_model"]!=embeddings_model:
        st.session_state["embeddings_model"]=embeddings_model
        st.session_state["embeddings_model_updated"]=True
        restart_assistant()
    if "rag_assistant" not in st.session_state or st.session_state["rag_assistant"] is None:
        st.session_state["rag_assistant"]=get_groq_assistant(
            llm_model=llm_model,
            embeddings_model=embeddings_model
        )
        st.session_state["messages"]=[]
    rag_assistant=st.session_state["rag_assistant"]
    if "url_scrape_key" not in st.session_state:
        st.session_state["url_scrape_key"]=0
    input_url=st.sidebar.text_input(
        "Add URL to Knowledge Base",key=st.session_state["url_scrape_key"]
    )
    if st.sidebar.button("Add URL"):
        if input_url:
            loader=WebBaseLoader(input_url)
            web_docs=loader.load()
            if web_docs:
                rag_assistant.retriever.vectorstore.add_documents(web_docs)
                st.sidebar.success(f"Added {len(web_docs)} documents from URL!")
            else:
                st.sidebar.error("Could not load documents from URL")
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"]=100
    uploaded_file=st.sidebar.file_uploader(
        "Add a PDF",type="pdf", key=st.session_state["file_uploader_key"]

    )
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pdf_docs=loader.load()
        if pdf_docs:
            rag_assistant.retriever.vectorstore.add_documents(pdf_docs)
            st.sidebar.success(f"Added {len(pdf_docs)} documents from PDF")
        else:
            st.sidebar.error("Could not read PDF")
    if "messages" not in st.session_state:
        st.session_state["messages"]=[]
    if prompt:=st.chat_input("Ask a question"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if st.session_state["messages"] and st.session_state["messages"][-1]["role"]=="user":
        last_msg=st.session_state["messages"][-1]["content"]
        with st.chat_message("assistant"):
            input_key = "question" if hasattr(rag_assistant, "retriever") else "input"
            result=rag_assistant({"question": last_msg})
            answer=result["answer"]
            st.write(answer)
            st.session_state["messages"].append({"role": "assistant","content": answer})
    if st.sidebar.button("New Run"):
        restart_assistant()
    if "embeddings_model_updated" in st.session_state:
        st.sidebar.info("Please add documents again as embeddings model changed")
        st.session_state["embeddings_model_updated"]=False
if __name__=="__main__":
    main()

