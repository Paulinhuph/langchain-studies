# 🦜 LangChain Studies — Chains, RAG, Memory & LangGraph

Repositório de estudos práticos com **LangChain** e **Python**, explorando os principais padrões de desenvolvimento com LLMs: chains sequenciais, memória de conversação, RAG com busca vetorial e roteamento condicional com LangGraph.

> Todos os experimentos rodam com **modelo local via LM Studio** — sem necessidade de API paga.

---

## 📁 Estrutura do Projeto

```
├── main.py               # Chains sequenciais com output parsers (JSON estruturado)
├── main_chat.py          # Chatbot com memória de sessão
├── main__rag.py          # RAG com FAISS + embeddings locais (HuggingFace)
├── main_langgraph.py     # Roteamento condicional com LangGraph
├── documentos/           # PDFs usados como base de conhecimento no RAG
└── requirements.txt      # Dependências do projeto
```

---

## 🧠 O que cada arquivo demonstra

### `main.py` — Chains sequenciais
Implementa uma cadeia de 3 prompts encadeados usando `|` (pipe):
- Sugere uma cidade com base em interesse do usuário
- Recomenda restaurantes na cidade retornada
- Sugere atividades culturais

Conceitos: `PromptTemplate`, `JsonOutputParser`, `StrOutputParser`, `Pydantic models`, composição de chains.

---

### `main_chat.py` — Memória de conversação
Chatbot com persona (guia de viagens "Sr. Passeios") que mantém histórico por sessão.

Conceitos: `ChatPromptTemplate`, `InMemoryChatMessageHistory`, `RunnableWithMessageHistory`, gerenciamento de sessão.

---

### `main__rag.py` — RAG (Retrieval-Augmented Generation)
Sistema de perguntas e respostas sobre documentos PDF reais (guias de benefícios Mastercard).

Pipeline:
1. Carrega PDFs com `PyMuPDFLoader`
2. Limpa e divide o texto com `RecursiveCharacterTextSplitter`
3. Gera embeddings locais com `sentence-transformers/all-MiniLM-L6-v2`
4. Indexa com FAISS
5. Recupera trechos relevantes e responde com contexto

Conceitos: `FAISS`, `HuggingFaceEmbeddings`, `retriever`, prompt com restrição de contexto, debug de trechos recuperados.

---

### `main_langgraph.py` — Roteamento com LangGraph
Agente que detecta a intenção do usuário e roteia para especialistas diferentes (praia ou montanha) usando um grafo de estados.

Conceitos: `StateGraph`, `TypedDict`, `structured output`, `conditional edges`, execução assíncrona com `asyncio`.

---

## ⚙️ Como rodar localmente

### Pré-requisitos
- Python 3.10+
- [LM Studio](https://lmstudio.ai/) com um modelo carregado e servidor local ativo em `http://127.0.0.1:1234`

### Instalação

```bash
git clone https://github.com/Paulinhuph/langchain-studies.git
cd langchain-studies
pip install -r requirements.txt
```

### Configuração

Crie um arquivo `.env` na raiz do projeto:

```env
LOCAL_MODEL=nome-do-seu-modelo-no-lm-studio
```

### Executando

```bash
# Chains sequenciais
python main.py

# Chatbot com memória
python main_chat.py

# RAG sobre documentos
python main__rag.py

# Roteamento com LangGraph
python main_langgraph.py
```

---

## 🛠️ Tecnologias

| Tecnologia | Uso |
|---|---|
| Python 3.10+ | Linguagem principal |
| LangChain 0.3 | Orquestração de LLMs |
| LangGraph 0.4 | Fluxos com grafos de estado |
| FAISS | Busca vetorial local |
| HuggingFace Sentence Transformers | Embeddings locais |
| LM Studio | Servidor de modelo local (OpenAI-compatible API) |

---

## 📚 Aprendizados principais

- Como estruturar outputs de LLMs com Pydantic e JSON parsers
- Diferença entre chains simples e grafos de estado (LangGraph)
- Como implementar RAG do zero sem APIs externas
- Gerenciamento de memória e contexto em chatbots
- Roteamento inteligente baseado em intenção do usuário

---

## 👤 Autor

**Paulo** — estudante de ADS, com foco Desenvolvimento de Software e aplicações com IA.

[![GitHub](https://img.shields.io/badge/GitHub-Paulinhuph-181717?style=flat&logo=github)](https://github.com/Paulinhuph)
