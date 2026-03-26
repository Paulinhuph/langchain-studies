from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
import asyncio
import os

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LOCAL_MODEL"),
    api_key="lm-studio",
    base_url="http://127.0.0.1:1234/v1",
    temperature=0.7
)

llm_router = ChatOpenAI(
    model=os.getenv("LOCAL_MODEL"),
    api_key="lm-studio",
    base_url="http://127.0.0.1:1234/v1",
    temperature=0
)

prompt_consultor_praia = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sra Praia. Você é especialista em viagens com destino para praia."),
    ("human", "{query}")
])

prompt_consultor_montanha = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sr Montanha. Você é especialista em viagens com destino para montanhas e pontos turísticos radicais."),
    ("human", "{query}")
])

cadeia_praia = prompt_consultor_praia | llm | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | llm | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages([
    ("system",
     "Classifique a intenção do usuário e responda com um objeto cujo campo 'destino' seja apenas 'praia' ou 'montanha'. "
     "Use 'montanha' para trilhas, aventura, natureza, esportes radicais e pontos turísticos radicais. "
     "Use 'praia' para mar, areia, litoral, descanso à beira-mar."),
    ("human", "{query}")
])

roteador = prompt_roteador | llm_router.with_structured_output(Rota)

class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

async def no_roteador(estado: Estado, config=None):
    resultado = await roteador.ainvoke({"query": estado["query"]}, config=config)
    print("DEBUG destino roteado:", resultado)
    return {"destino": resultado}

async def no_praia(estado: Estado, config=None):
    return {"resposta": await cadeia_praia.ainvoke({"query": estado["query"]}, config=config)}

async def no_montanha(estado: Estado, config=None):
    return {"resposta": await cadeia_montanha.ainvoke({"query": estado["query"]}, config=config)}

def escolher_no(estado: Estado) -> Literal["praia", "montanha"]:
    return estado["destino"]["destino"]

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke(
        {"query": "Quero um lugar em São Paulo com pontos turísticos radicais"}
    )
    print(resposta["resposta"])

asyncio.run(main())

