# Building a RAG Application with Streamlit, ChromaDB, LangChain, and OpenAI

This guide walks you through creating a RAG (Retrieval-Augmented Generation) application using Streamlit.  
The workflow:
1. Load external data from URLs.  
2. Embed and store the content in **ChromaDB**.  
3. Use **LangChain** to retrieve relevant chunks.  
4. Pass the results into **OpenAI LLM** for answering user queries.  

---

Pre-requisites:
1. Have the OpenAI key in the .env
2. Do `pip install -r requirements.txt`
