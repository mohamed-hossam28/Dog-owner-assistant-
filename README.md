🐾 Dog Owner Assistant — Agentic RAG + WebSearch Chatbot
An intelligent veterinary assistant powered by multi-agent Retrieval-Augmented Generation and live web search fallback.
Created as a graduation project for the NTI Summer Training Program.

🚀 Project Overview
This interactive Streamlit app helps dog owners get reliable health and care information. It cleverly combines:

RAG (Retrieval-Augmented Generation): Answers your question using an internal veterinary knowledge base.
Web Search Agent: If the knowledge base can’t answer, automatically searches the web for the latest information.
Transparent Sourcing: Tells you clearly whether the answer came from trusted reference materials or from online sources.
Developed during my time in the NTI Summer Training Program, this project demonstrates modern GenAI, agentic reasoning, and practical deployment skills.

✨ Features
🤖 Smart Routing: Multi-agent LangChain architecture intelligently selects between internal knowledge base and live web.
🌐 Always Up-To-Date: If new veterinary questions aren’t in the database, you get real-time web answers.
🪪 Source Disclosure: Each answer is clearly labeled as coming from “Internal Knowledge Base” or “Web Search.”
🖥️ User-Friendly UI: Simple Streamlit interface for anyone to use.
🔐 .env Key Management: Secure storage of API keys for LLM and search.
🛠️ Tech Stack
Python 3.9+
Streamlit
LangChain
OpenRouter API (Mixtral 8x7b-instruct model)
ChromaDB (for vector storage)
HuggingFace Sentence-Transformers
Tavily Web Search
dotenv (for env config)

🚦 Installation:
Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dog-owner-assistant.git
cd dog-owner-assistant
```

Install dependencies
```bash
pip install streamlit langchain langchain-openai langchain-community chromadb sentence-transformers tavily-search python-dotenv
```

Create your .env file
```text
OPENAI_API_KEY=your_openrouter_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

(Optional) Prepare your vectorstore
Build your chroma_db folder with internal vet knowledge (see data/ or relevant scripts if included).

Run the app
```bash
streamlit run app.py
```

🐕 Example Use
Question: "How do I treat parvovirus in puppies?"
Agent: Finds the answer in the knowledge base or, if missing, queries the web.
You See:
text
Answer source: Internal Knowledge Base 🤓
<The factual answer>
