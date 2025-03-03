from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from typing import List


class Gemini:
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self._llm_model = ChatVertexAI(
            model_name=self.model_name,
            temperature=self.temperature,
        )

    @property
    def llm_model(self):
        return self._llm_model

    def rag_chain(self, prompt: ChatPromptTemplate, retriever):
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm_model
            | StrOutputParser()
        )
