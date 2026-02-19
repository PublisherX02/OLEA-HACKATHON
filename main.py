import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.schema import Document
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in environment variables")

@dataclass
class InssuranceChatbotConfig:
    model_name: str = "meta/llama-3.1-70b-instruct"  
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    temperature: float = 0.2
    max_tokens: int = 1024
    k_documents: int = 4

config = InssuranceChatbotConfig()


#Database Setup using RAG documents


llm = ChatNVIDIA(
    model=config.model_name,
    api_key=NVIDIA_API_KEY,
    temperature=config.temperature,
    max_tokens=config.max_tokens
)

embeddings = HuggingFaceEmbeddings(
    model_name=config.embedding_model,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Create Document objects
Insurance_documents = [] # Placeholder: Documents will be loaded here later
docs = [Document(page_content=doc) for doc in Insurance_documents] #documents and informations will e rovided later
splits = text_splitter.split_documents(docs)

print(f"✅ Split {len(docs)} documents into {len(splits)} chunks")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="banking_knowledge"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": config.k_documents}
)

Insurance_prompt_template = """
You are an expert Insurance...






Context from knowledge base:
{context}

Customer Question: {question}

Your Response:
"""

INSURANCE_PROMPT = PromptTemplate(
    template=Insurance_prompt_template,
    input_variables=["context", "question"]
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": INSURANCE_PROMPT},
    return_source_documents=True
)

print("✅ RAG chain created successfully")

def query_rag(question: str) -> Dict[str, Any]:
    """Query the RAG system and return results with sources"""
    result = rag_chain({"query": question})
    return {
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }

#agents config
#name a dictionary named tools containing each tool
tools = [] # Placeholder: Tools will be defined later


agent_prompt_template = """
You are a helpful Insurance assistant with access to various tools. 
You can help customers with account information, transfers, Insurance policies, and general banking questions.

IMPORTANT:
- Always verify account IDs before performing operations
- For Money transactions, confirm the amount and accounts before executing
- Be clear and professional in all responses
- Use the KnowledgeBase tool for general Insurance questions
- Use specific tools for account operations

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

agent_prompt = PromptTemplate(
    template=agent_prompt_template,
    input_variables=["input", "agent_scratchpad"],
    partial_variables={
        "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        "tool_names": ", ".join([tool.name for tool in tools])
    }
)

# Create agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt
)

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)


class InsuranceChatbot:
    """Complete banking chatbot with RAG and Agents"""
    
    def __init__(self, agent_executor, rag_chain):
        self.agent_executor = agent_executor
        self.rag_chain = rag_chain
        self.conversation_history = []
    
    def chat(self, user_input: str, use_agent: bool = True) -> Dict[str, Any]:
        """
        Main chat interface
        
        Args:
            user_input: User's question or request
            use_agent: If True, use agent for complex tasks; if False, use RAG only
        
        Returns:
            Dictionary with response and metadata
        """
        timestamp = datetime.now().isoformat()
        
        try:
            if use_agent:
                # Use agent for complex operations
                response = self.agent_executor.invoke({"input": user_input})
                answer = response["output"]
                mode = "agent"
            else:
                # Use RAG for simple Q&A
                response = query_rag(user_input)
                answer = response["answer"]
                mode = "rag"
            
            # Store in conversation history
            interaction = {
                "timestamp": timestamp,
                "user_input": user_input,
                "response": answer,
                "mode": mode
            }
            self.conversation_history.append(interaction)
            
            return {
                "success": True,
                "response": answer,
                "mode": mode,
                "timestamp": timestamp
            }
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            return {
                "success": False,
                "response": error_msg,
                "error": str(e),
                "timestamp": timestamp
            }
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history[-limit:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.agent_executor.memory.clear()
        print("✅ Conversation history cleared")

# Initialize chatbot
chatbot = InsuranceChatbot(agent_executor, rag_chain)













