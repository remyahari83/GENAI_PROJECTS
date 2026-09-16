"""
LangChain_Multiroute_customer_Support.py (Modern LangChain Version)
------------------------------------------------
Concept

This project demonstrates how to use LangChain's RunnableBranch to implement conditional routing. Based on the type of user query, the application routes the request to the appropriate response chain.

Purpose

The project simulates a customer support assistant with three different conversation paths:

Product Query → Routes to the Product Chain for questions about cloud products, plans, pricing, and features.
Technical Support Query → Routes to the Technical Support Chain for DevOps, Kubernetes, cloud infrastructure, deployment, and troubleshooting issues.
General Query → Routes to the General Chain for questions that don't belong to the above categories.

Example:
Given a user query, the system decides:
  1️⃣ If it's a question about cloud product → use the "Product" response chain.
  2️⃣ If it's a technical question → respond with technical guidance chain.
  3️⃣ Otherwise query is related to general question - use generic response chain.
"""

import os
import json
from dotenv import load_dotenv

# Silence logs
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda


# ==========================================
# 1️⃣ Setup
# ==========================================
load_dotenv()

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
cfg = config[provider]

# Initialize model
if provider == "openai":
    llm = ChatOpenAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 250),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.7),
        max_output_tokens=cfg.get("max_output_tokens", 250),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

# ==========================================
# Setup Summarizing Message History
# ==========================================
# from langchain.memory import ConversationSummaryBufferMemory  # temporary utility
# # Note: RunnableWithMessageHistory doesn't yet auto-summarize; we'll mimic it here.
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InmemoryConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
    return store[session_id]

# ==========================================
# 2️⃣ Define Branch-Specific Chains
# ==========================================

# Branch 1: Product Expert Response
product_prompt = ChatPromptTemplate.from_template(
    """
You are a cloud product expert.

Previous conversation:
{chat_history}

Current question:
{query}

Answer the current question using the previous conversation when relevant.
"""
)
product_chain = product_prompt | llm | StrOutputParser()

# Branch 2: Greeting Response
tech_prompt = ChatPromptTemplate.from_template(
    """
    You are a technical support engineer specializing in DevOps,
    Kubernetes and cloud infrastructure.
    
    Previous conversation:
    {chat_history}
    
    Current question:
    {query}
    
    Provide clear troubleshooting guidance.
    """
)
tech_chain = tech_prompt| llm | StrOutputParser()

# Branch 3: Generic Fallback
generic_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Previous conversation:
{chat_history}

Current question:
{query}

Answer clearly.
""")
   
generic_chain = generic_prompt | llm | StrOutputParser()


# ==========================================
# 3️⃣ Define Conditional Logic Function
# ==========================================
def route_input(x):
    query = x["query"].lower()

    if any(word in query for word in ["price", "product", "plan", "feature"]):
        return "PRODUCT"

    if any(word in query for word in [
        "kubernetes", "pod", "docker", "terraform",
        "deployment", "error", "crashloopbackoff"
    ]):
        return "SUPPORT"

    return "GENERAL"
# ==========================================
# 4️⃣ Combine with RunnableBranch
# ==========================================
router_chain = RunnableBranch(
    # Condition → Chain pairs
    (lambda x: route_input(x) == "PRODUCT", product_chain),
    (lambda x: route_input(x) == "SUPPORT", tech_chain),
    # Default branch
    generic_chain,
)

summary_prompt = ChatPromptTemplate.from_template("""
Summarize the following conversation.

Keep important information such as:
- user's problem
- products mentioned
- technical details
- previous solutions
- important user requirements

Conversation:
{conversation}

Return a concise summary.
""")

summary_chain = summary_prompt | llm | StrOutputParser()
# ==========================================
# 5️⃣ Interactive CLI Demo
# ==========================================
print("\n🔀 RUNNABLE BRANCH DEMO — Conditional Logic Flow")
print("Routes user queries intelligently to different chains.\n")
print("----------------------------------------")

conversation = []
summary = ""

while True:

    user_input = input("💬 Enter your message ('exit' to quit): ").strip()

    if user_input.lower() in ["exit", "quit"]:
        break

    # Send query + summary to RunnableBranch
    result = router_chain.invoke({
        "query": user_input,
        "chat_history": summary
    })

    print(f"🤖 Assistant: {result}\n")

    # Store conversation
    conversation.append({
        "role": "user",
        "content": user_input
    })

    conversation.append({
        "role": "assistant",
        "content": result
    })

    # Create updated summary
    conversation_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation
    )

    summary = summary_chain.invoke({
        "conversation": conversation_text
    })
print("""
----------------------------------------
📘 Key Takeaways:
1️⃣ RunnableBranch allows conditional routing between chains.
2️⃣ Each branch can represent a different logic or LLM prompt.
3️⃣ It's the modern replacement for RouterChain (cleaner & flexible).
""")
