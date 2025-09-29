import streamlit as st

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

st.title("RAG App")

urls = [
    "https://www.hypernode.com/en/",
    "https://www.hypernode.com/en/for-agencies/",
    "https://www.hypernode.com/en/managed-cloud-hosting/",
    "https://www.hypernode.com/en/managed-dedicated-hosting/",
    "https://www.hypernode.com/en/cluster-hosting/",
    "https://www.hypernode.com/en/plans-and-prices/",
    "https://www.hypernode.com/en/performance-features/",
    "https://www.hypernode.com/en/security/",
    "https://www.hypernode.com/en/managed-ssl-certificates/"

]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = OpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=200)

query = st.chat_input("Ask your question: ")

system_prompt = (
    "This is a question-asnwering task."
    "Use the following pieces of retrieved context for your answers."
    "If you don't know the answer, say I don't know."
    "Use maximum 3 sentences for the answers."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])