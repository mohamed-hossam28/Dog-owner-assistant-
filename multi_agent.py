import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain.tools import Tool
from langchain.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType

# --- Configuration and API keys ---
def configure():
    load_dotenv()  # Load environment variables from .env file

configure()

# Set API keys (adjust variable names as needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("api_key")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing! Please add it to your .env file.")
    st.stop()
if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY missing! Please add it to your .env file.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# --- Initialize LLM ---
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    model="mistralai/mixtral-8x7b-instruct",
    temperature=0
)

# --- Initialize embeddings and vectorstore ---
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"normalize_embeddings": True, "device": "cpu"},
        model_kwargs={"device": "cpu"}
    )
except Exception as e:
    st.error(f"Error initializing embeddings: {str(e)}")
    embeddings = None

CHROMA_PATH = "chroma_db"

def load_vectorstore(chroma_path):
    if embeddings is None:
        st.error("Embeddings model could not be initialized. Cannot load vector store.")
        return None
    try:
        return Chroma(
            collection_name="rag_store",
            embedding_function=embeddings,
            persist_directory=chroma_path
        )
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
        return None

try:
    vectorstore = load_vectorstore(CHROMA_PATH)
except Exception as e:
    st.error(f"Error in vector store initialization: {str(e)}")
    vectorstore = None

def get_relevant_documents(query):
    if vectorstore is None:
        return []
    try:
        return vectorstore.similarity_search_with_score(query, k=3)
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

# --- Build RAG prompt ---
def build_prompt(kwargs):
    docs = kwargs["context"]
    question = kwargs["question"]

    if docs:
        try:
            context = "\n\n".join(str(doc[0].page_content) for doc in docs)
        except Exception as e:
            st.error(f"Error processing document content: {str(e)}")
            context = "Error retrieving context information."
    else:
        context = "No relevant information found."

    prompt_template = f"""You are a veterinary assistant AI helping with animal health questions.

CONTEXT INFORMATION:
{context}

INSTRUCTIONS:
- Answer the user's question ONLY using information from the context provided above.
- If the context doesn't contain information to answer the question, respond with: "I don't have enough information in my reference materials to answer this question properly. Please consult with a veterinarian for specific advice about your pet."
- Do not make up or infer information that isn't explicitly stated in the context.
- Keep your answers factual and based solely on the provided veterinary reference materials.
- Format your response clearly and professionally.

USER QUESTION: {question}

YOUR ANSWER:"""

    formatted_prompt = prompt_template.format(context=context, question=question)
    return HumanMessage(content=formatted_prompt)

SIMILARITY_THRESHOLD = 0.7   

def run_rag_tool(question):
    context_docs = get_relevant_documents(question)
    
    if not context_docs:
        return "I don't have enough information in my reference materials to answer this question properly. Please consult with a veterinarian for specific advice about your pet."

    if all(score > SIMILARITY_THRESHOLD for (_, score) in context_docs):
        return "I don't have enough information in my reference materials to answer this question properly. Please consult with a veterinarian for specific advice about your pet."
    prompt_message = build_prompt({"context": context_docs, "question": question})
    response = llm([prompt_message])
    return response.content

tavily_search = TavilySearchResults(api_key=os.environ["TAVILY_API_KEY"])

def run_web_search(question):
    try:
        results = tavily_search.run(question)
        summary = "\n".join(
            f"{r.get('title','')}: {r.get('snippet','')}" for r in results[:3]
        )
        return summary or "Sorry, I could not find a relevant answer on the web."
    except Exception as e:
        return f"Web search tool error: {str(e)}"

rag_tool = Tool(
    name="InternalRAG",
    func=run_rag_tool,
    description="Use this tool to answer dog health questions from the veterinary reference material, if enough relevant context is found."
)
web_tool = Tool(
    name="WebSearch",
    func=run_web_search,
    description="Use this for recent, novel, or uncovered dog health/care questions."
)

system_message = """
You are a veterinary AI assistant.
- Always use 'InternalRAG' first. If the InternalRAG response contains "I don't have enough information", then use 'WebSearch'.
- Use 'WebSearch' for anything not covered in the reference material that is  represented in vectorstore.
Be explicit and only use information from the selected tool and say where it from the reference material or the web search  .
"""

router_agent = initialize_agent(
    [rag_tool, web_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    agent_kwargs={'system_message': system_message},
    return_intermediate_steps=True
)

# --- Streamlit App ---
def main():
    st.title("🐾 Dog Owner Assistant (Agentic RAG)")
    st.subheader("Ask questions about Dog health and care")

    if vectorstore is None:
        st.warning("No knowledge base is loaded. The assistant will rely on web search.")

    user_input = st.text_area("Ask a veterinary question:", height=100)

    if st.button("Submit", type="primary"):
        if user_input:
            with st.spinner("Thinking..."):
                try:
                    st.info("Routing your question using a multi-agent system.")
                    agent_result = router_agent(user_input)
                    answer = agent_result['output']
                    intermediate_steps = agent_result['intermediate_steps']

                    if intermediate_steps:
                        tool_used = intermediate_steps[-1][0].tool  # Last tool run
                        if tool_used == "InternalRAG":
                            label = "**Answer source: Internal Knowledge Base 🤓**"
                        elif tool_used == "WebSearch":
                            label = "**Answer source: Web Search 🌐**"
                        else:
                            label = "**Answer source: Unknown**"
                    else:
                        label = "**Answer source: (undetermined)**"

                    st.markdown(label)
                    st.markdown("### Response:")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")

# Add some styling
st.markdown("""
<style>
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()