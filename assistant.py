
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import asyncio
from langchain.embeddings import HuggingFaceEmbeddings
load_dotenv()
from langchain.docstore import InMemoryDocstore
from typing import List
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
async def create_embeddings(embeddings_model: str):
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
async def create_llm(llm_model: str):
    return ChatGoogleGenerativeAI(temperature=0.0, model=llm_model)
def get_groq_assistant(
        llm_model: str="gemini-1.5-flash",
        embeddings_model: str="text-embedding-3-small",
        docs: List[Document]=None
):
    llm = asyncio.run(create_llm(llm_model))
    embeddings = asyncio.run(create_embeddings(embeddings_model))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key="answer")
    if docs:
        vectorstore=FAISS.from_documents(docs,embeddings)


    else:

        # get embedding dimension
        dummy_vector = embeddings.embed_query("dummy text")
        vector_dim = len(dummy_vector)

        # create an empty FAISS index
        import faiss
        index = faiss.IndexFlatL2(vector_dim)
        vectorstore = FAISS(
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            embedding_function=embeddings.embed_query,
        )
    conv_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                       retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
                                                       memory=memory, return_source_documents=True,output_key="answer"
                                                       )

    return conv_chain
