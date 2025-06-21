import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

groq_api = os.getenv('GROQ_API_KEY')

from langchain_groq import ChatGroq

llm = ChatGroq(model='Gemma2-9b-It', groq_api_key = groq_api)

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a celebrity bot that answers questions about celebrities. "
    ),
    HumanMessage(
        content=" Who is the tribal chief in wwe? "
    )
]

response = llm.invoke(messages)

print(response.content)

#parse the output

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
response_parsed = parser.invoke(response)

#LCEL 

chain = llm|parser
response_chain = chain.invoke(messages)
print(response_chain)

# Prompt Template

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a QA bot that answers questions about {celebrity}."),
        ("human", "{question}"),
    ]
)

response_prompt = prompt.invoke({"celebrity": "The Rock", "question": "Who is the tribal chief in wwe?"})
print(response_prompt)

chain = prompt|llm|parser
response_chain = chain.invoke({"celebrity": "The Rock", "question": "Who is the tribal chief in wwe?"})
print(response_chain)