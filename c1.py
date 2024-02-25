from langchain.chat_models import ChatOpenAI
from langchain.schema import ( AIMessage, HumanMessage, SystemMessage)
import os
from dotenv import load_dotenv 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

load_dotenv()
print(os.getenv ("OPENAI_API_KEY"))


Chat = ChatOpenAI(temperature=0.9)

# messages = [ SystemMessage(content = "your are helpful chat assistant, Answer my question")]

# while True: 
#     user_input = input(">")
#     print("Message Hist")
#     print(messages)
#     print("--------")
#     messages.append(HumanMessage(content="How is weather in Dallas during Febravery? "))
#     airesponse = Chat(messages)
#     messages.append(AIMessage(content=airesponse.content))
#     print(airesponse)
    

conversation = ConversationChain ( llm=Chat, memory=ConversationBufferMemory(memory_key="history",return_messages=True), verbose=True)

while True: 
    user_input = input(">")
    ai_response = conversation.predict(input = user_input)
    print(conversation)