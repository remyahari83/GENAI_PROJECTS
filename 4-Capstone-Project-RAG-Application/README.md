Streamlit application with RAG architecture with multi-llm chat, document uploads, querying and multichatsupport.
Data can come from pdf, text or web and will be uploaded(loader.py) and split(splitter.py). Embedding(embedding.py) module will embed these split documents and updates the vectore store (FAISS/pinecone). 
The query from user is embedded , relevant chunks are retrieved by retriver(retriever.py) passed to llm along with context. Response is generated using a rag_chain rag_chain.py.

Execution:
1. From your project directory , execute below to create venv:
py -3.11 -m venv .venv

2. Activate it, from powershell:
.\.venv\Scripts\Activate.ps1

3. Verify python version:
python --version
Desired version -Python 3.11.x

4. Then upgrade pip:
python -m pip install --upgrade pip

5. Then install your requirements:
python -m pip install -r requirements.txt

6. Execute the Application.
streamlit run app/streamlit_aop.py
