# Intelligent-Complaint-Analysis
This repository contains the codebase and project structure for building an AI-powered complaint analysis tool for CrediTrust Financial, a digital finance company operating in East Africa.

The system empowers internal teams—Product, Support, Compliance, and Executives—to ask natural language questions and receive evidence-backed summaries from real customer complaints, enabling:

● Faster trend identification

● Proactive issue resolution

● Enhanced decision-making across 5 key financial products:

    ● Credit Cards

    ● Personal Loans

    ● Buy Now, Pay Later (BNPL)

    ● Savings Accounts

    ● Money Transfers

## 🔑 Key Features

● Plain-English question answering from complaint data

● Semantic search with FAISS or ChromaDB

● LLM-generated responses backed by retrieved complaint narratives

● Real-time analysis across product categories

● Semantic search over complaint chunks

● Retrieve top relevant complaint texts based on user queries

● Generate natural-language answers using a local LLM (e.g., facebook/opt-1.3b)

● Dashboard or chatbot interface for internal users

## 📊 Project Goals & KPIs

● Reduce time to identify major complaint trends from days to minutes

● Enable non-technical teams to self-serve insights

● Shift from reactive to proactive issue management

## ⚙️ Technologies Used

● Python, LangChain, FAISS/ChromaDB

● OpenAI/GPT/LLM APIs

● Streamlit (for UI) or Chatbot integration (e.g., Rasa, Gradio)

● Pandas, spaCy, scikit-learn (for preprocessing and analytics)

● SentenceTransformers for embedding complaint texts

● FAISS for fast similarity search

● transformers (LLM) for generating context-aware answers

🧠 LLM Pipeline:

Embeddings generated using "all-MiniLM-L6-v2"

Local language model used for answer generation (can be swapped for API-based or smaller models)