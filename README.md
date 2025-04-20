
# 🚀 AI-Powered RAG Pipeline for Multimodal Applications

A modular and scalable AI pipeline built for **Retrieval-Augmented Generation (RAG)**  with **multimodal capabilities** to handle both text and image inputs. Designed for flexibility, this solution integrates seamlessly with various **LLMs**, **vector stores**, and **custom applications**, making it adaptable for a wide range of use cases.

---

## 🌟 Key Features

- **🔧 Modular Configuration**  
  Easily customizable via YAML-based config files. Quickly switch between models, services, or settings — perfect for experimenting or scaling to new domains.

- **🧠 Multi-LLM Support**  
  Plug-and-play with local and cloud-based LLMs like **Phi4 via Ollama** or **OpenAI’s GPT-3/4**, depending on your needs and resources.

- **📦 Pluggable Vector Stores**  
  Supports both open-source and cloud vector databases like **FAISS** and **Pinecone** for efficient document retrieval and similarity search.

- **⚙️ FastAPI-Based Modular Apps**  
  Components like vector store loaders, RAG logic, and model connectors are modularized and wrapped in clean FastAPI apps — easy to extend, reuse, or deploy.

- **🔐 Environment-Safe**  
  Sensitive credentials are securely managed via environment variables — no hardcoding.

- **🖼️ Multimodal Capabilities**  
  Started as a text-only RAG pipeline, now extended to support **LLaVA** for tasks involving both **image and text inputs** — enabling advanced multimodal querying.

---

## 🛠️ Tech Stack

- **Languages**: Python  
- **Core Libraries**: OpenAI API, Ollama, Sentence Transformer,FastAPI, FAISS, Pinecone, PyYAML  
- **AI Models**: LLaVA, Phi4, GPT-3/4  
- **Deployment**: FastAPI  
- **Cloud Services**: Pinecone (vector storage)

---
