import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel


load_dotenv()

class Destino(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    motivo:str = Field("motivo pelo qual é interessante visitar essa cidade")


class Restaurantes(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    restaurantes:str = Field("Restaurantes recomendados na cidade")


parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    input_variables=["interesse"],
    template="""Sugira uma cidade dado o meu interesse por {interesse}. {formato_de_saida}""",
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()}
)

prompt_restaurantes = PromptTemplate(
    template="""Sugira restaurantes populares entre locais em {cidade}. {formato_de_saida}""",
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template="Sugira atividades e locais em {cidade}"
)



llm = ChatOpenAI(
    model=os.getenv("LOCAL_MODEL"),
    api_key="lm-studio",
    base_url="http://127.0.0.1:1234/v1",
    temperature=0.7
)

cadeia_1 = prompt_cidade | llm | parseador_destino
cadeia_2 = prompt_restaurantes | llm | parseador_restaurantes
cadeia_3 = prompt_cultural | llm | StrOutputParser()

from langchain_core.runnables import RunnableLambda

cadeia = (
    cadeia_1 
    | cadeia_2 
    | RunnableLambda(lambda x: {"cidade": x["cidade"]})
    | cadeia_3
)

print("Prompt:", prompt_cidade)
resposta = cadeia.invoke({"interesse": "praias"})
print(resposta)