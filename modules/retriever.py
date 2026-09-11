from langchain.retrievers import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from modules.utils import get_embedding_model
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def get_retriever(vectorstore, config):

    retriever_type = config.get("retriever_type", "base").lower()

    if retriever_type == "base":
        print("Using base retriever(simple similarity search)")
        return vectorstore.as_retriever(search_kwargs = {"k": 3})
    elif retriever_type == "multiquery":
        print("Using multi-query retriever...")
        llm_provider = config.get("llm_provider", "openai").lower()

        if llm_provider == "openai":
            llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
        elif llm_provider == "gemini":
            llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        return MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs = {"k": 3}),
            llm = llm
        )
    else:
        raise ValueError(f"Unsupported retriever_type: {retriever_type}")