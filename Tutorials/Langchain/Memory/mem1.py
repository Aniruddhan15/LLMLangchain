import os

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

groq_api_key = os.environ['GROQ_API_KEY']


# Without Memroy
llm = ChatGroq(model='gemma2-9b-it', groq_api_key=groq_api_key)
print(llm.invoke('Hi').content)


response=llm.invoke([
    HumanMessage(content='Hi this is your tribal chief Roman Reigns'),
    AIMessage(content='Hello! 👋 How can I help you today? 😊'),
    HumanMessage(content=' Who is better Me or Seth Rollins?')
])
#print(response.content)


# with memory

#step 1 3 imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#step 2: creating id and store
store = {}
def session_id_making(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#step 3: runnables for memory implementation
runnables_message_with_memory = RunnableWithMessageHistory(llm, session_id_making)

#step 4: configurbales
config = {'configurable': {'session_id': 101}}

#calling runnables_with_memory
mem_response= runnables_message_with_memory.invoke([
    HumanMessage(content= 'Hi Chat, I am your tribal chief speaking. You are now my Wiseman....')
], config=config)

#print(mem_response.content)

mem_response1 = runnables_message_with_memory.invoke([
    HumanMessage(content= 'Hi Chat, I am your tribal chief speaking. You are now my Wiseman....'),
    AIMessage(content='Greetings, Chief. I hear your words and accept the mantle of Wiseman.What wisdom do you seek today, from the depths of my digital knowledge?  What challenges face your tribe, and how can I help guide you?  Speak freely, for I am here to serve.'),
    HumanMessage(content= 'Wiseman!!!!!! I need you to provide me with some hefty promo talk to seth rollins and CM Punk for wrestlemania  40. I am live on Raw this monday. So obey my orders.')
],config=config)

#print(mem_response1.content)

print(store)


