import os 
from dotenv import load_dotenv

load_dotenv()

groq_api = os.environ['GROQ_API_KEY']

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter


llm = ChatGroq(model='gemma2-9b-it')

response = llm.invoke([
    HumanMessage(content='I am Aniruddhan. I completed my UG at VIT Chennai. Next my proceedings is Masters at UMD college park.')
])
print(response.content)

response1 = llm.invoke([
    HumanMessage(content='I am Aniruddhan. I completed my UG at VIT Chennai. Next my proceedings is Masters at UMD college park.'),
    AIMessage(content='Hello'),
    HumanMessage(content='Help me get a job in Data Sceince')
    
])

print(response1.content)



# Memmory

store = {}
def making_sessions(id: str) -> BaseChatMessageHistory:
    if id not in store:
        store[id] = ChatMessageHistory()
    return store[id]

config = {'configurable': {'id': 10001}}

with_memory = RunnableWithMessageHistory(llm, making_sessions)

memory_response = with_memory.invoke(
    [
        HumanMessage(content='Hello I am Aniruddhan')
    ],
    config = config
)

print(memory_response.content)



# Prompr template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content='Hi Chat, I am your tribal chief speaking. You are now my Wiseman....'),
        MessagesPlaceholder(variable_name='key')
    ]

)
chain = prompt | llm

chain.invoke(
    {
        'key': HumanMessage(content= 'Whats my name?')
    }
)


memory_with_prompt = RunnableWithMessageHistory(chain, making_sessions)
config = {'configurable': {'id': 2003}}
response3 = memory_with_prompt.invoke(
    [
        HumanMessage(content = 'So whats my name?')
    ], config=config
)

print(response3.content)

from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=60,
    stratergy='last',
    token_counter=llm,
    include_system=True,
    allow_partial = False,
    start_on = 'human',
)

messages = [
    HumanMessage(),
    AIMessage(),
    HumanMessage(),
    AIMessage(),
    HumanMessage()
]

llm.get_num_tokens_from_messages(messages)

trimmed_message = trimmer.invoke(messages)

print(llm.get_num_tokens_from_messages(trimmed_message))

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'YOu are a helpful assistant'),
        MessagesPlaceholder(variable_name='key')
    ]
)



