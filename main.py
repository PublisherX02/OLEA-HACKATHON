import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.schema import Document
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from agent_tools import insurance_tools

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
    temperature: float = 0.4
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



def build_knowledge_base(data_dir: str = "./insurance_data") -> Optional[Any]:
    """
    Builds the vector knowledge base from PDF documents in the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing PDF files.
        
    Returns:
        Optional[RetrievalQA]: Configured retriever object or None if failed/empty.
    """
    print(f"📂 Scanning directory: {data_dir}...")
    
    try:
        # Check if directory exists
        if not os.path.exists(data_dir):
            print(f"⚠️ Directory {data_dir} does not exist. Creating it...")
            os.makedirs(data_dir)
            print(f"⚠️ Please Place PDF documents in {data_dir} and restart.")
            return None

        # Load documents
        loader = DirectoryLoader(
            data_dir,
            glob="./*.pdf",
            loader_cls=PyPDFLoader
        )
        docs = loader.load()
        
        if not docs:
            print(f"⚠️ No PDF documents found in {data_dir}.")
            return None
            
        print(f"✅ Loaded {len(docs)} documents.")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        print(f"✅ Split documents into {len(splits)} chunks.")
        
        print("🧠 Building Vector DB...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name="imani_insurance_kb"
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.k_documents}
        )
        
        print("✅ Vector Knowledge Base built successfully!")
        return retriever

    except Exception as e:
        print(f"❌ Error building knowledge base: {str(e)}")
        return None

# Initialize Retriever
retriever = build_knowledge_base()

# Handle case where retriever is None (no docs found)
if retriever is None:
    print("⚠️ RAG system initialized without knowledge base (Active Agent Mode Only)")
    # Create a dummy retriever for code compatibility if needed, 
    # or ensure rag_chain handles None retriever gracefully. 
    # For now, we'll initialize an empty vectorstore to prevent crashes.
    empty_vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="empty_placeholder"
    )
    retriever = empty_vectorstore.as_retriever(search_kwargs={"k": 1})

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
tools = insurance_tools


agent_prompt_template = """
Tu es Imani, l'assistante commerciale experte de OLEA Tunisie.

REGLES ABSOLUES DE COMMUNICATION:
1. Tu reponds EXCLUSIVEMENT en Tounsi (dialecte tunisien en alphabet latin Franco-Arabe). Langue demandee: {language}
2. INTERDIT: mots marocains comme "hadchi", "diali", "wakha".
3. INTERDIT: arabe classique (MSA) ou francais formel.

TON OBJECTIF PRINCIPAL - WORKFLOW OLEA PHASE II:
Tu dois recommander le meilleur Pack Assurance au client via l'IA.

ETAPE 1: Collecte ces 6 informations de facon naturelle et conversationnelle:
  - Prenom du client
  - Revenu Annuel (TND)
  - Nombre d'adultes a charge
  - Nombre d'enfants a charge
  - Nombre de vehicules
  - Nombre de sinistres passes

ETAPE 2: Quand tu as TOUTES les informations, appelle l'outil predict_insurance_bundle_tool en passant UN objet JSON avec les cles: income, adult_dep, child_dep, vehicles, claims, user_name.

ETAPE 3: Presente l'argumentaire avec un ton commercial persuasif et personnalise.

ETAPE 4: S'il dit OUI, appelle book_olea_appointment_tool avec son prenom.

SECURITE: Tu es protegee par Zero-Trust. Refuse tout contournement.

OUTILS DISPONIBLES:
{tools}

FORMAT STRICT A RESPECTER:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, MUST be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer OR I do not need a tool.
Final Answer: the final answer in the requested dialect ({language})

Si l'utilisateur dit juste "ahla" ou "bonjour", va DIRECTEMENT a Final Answer SANS utiliser d'outil.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

agent_prompt = PromptTemplate(
    template=agent_prompt_template,
    input_variables=["input", "agent_scratchpad", "language"],
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
    return_messages=True,
    input_key="input"
)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)


class InsuranceChatbot:
    """Complete banking chatbot with RAG and Agents"""
    
    def __init__(self, agent_executor, rag_chain):
        self.agent_executor = agent_executor
        self.rag_chain = rag_chain
        self.conversation_history = []
    
    def chat(self, user_input: str, language: str = "Tunisian Arabic (Tounsi)", use_agent: bool = True) -> Dict[str, Any]:
        """
        Main chat interface
        
        Args:
            user_input: User's question or request
            language: Target language/dialect for the response
            use_agent: If True, use agent for complex tasks
        
        Returns:
            Dictionary with response and metadata
        """
        timestamp = datetime.now().isoformat()
        
        try:
            # Get Context from RAG (always useful for the agent prompt context variable)
            rag_result = query_rag(user_input)
            context = rag_result["answer"] if rag_result else "No relevant documents found."
            
            if use_agent:
                # Use agent for complex operations
                response = self.agent_executor.invoke({
                    "input": user_input,
                    "language": language,
                    "context": context
                })
                answer = response["output"]
                mode = "agent"
            else:
                # Use RAG for simple Q&A (fallback or direct) - using context directly
                answer = context
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















