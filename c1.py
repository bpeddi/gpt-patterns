from langchain.chat_models import ChatOpenAI
from langchain.schema import ( AIMessage, HumanMessage, SystemMessage)
import os
from dotenv import load_dotenv 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from openai import OpenAI

load_dotenv()
print(os.getenv ("OPENAI_API_KEY"))


def chat_with_openai (model,messages) : 
    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=True
                    )
    return chat_completion

question = "How to make chiken curry?"
prompt = [{"role": "user", "content": question}]
response = []
result = ""
# st.write ( "Sending following Message", prompt)
for chunk in chat_with_openai(
    model="gpt-3.5-turbo", messages=prompt
    ):
    # print(chunk)
    text = chunk.choices[0].delta.content
    if text is not None:
        response.append(text)
        result = "".join(response).strip()
print(result)


# Chat = ChatOpenAI(temperature=0.9)

# # messages = [ SystemMessage(content = "your are helpful chat assistant, Answer my question")]

# # while True: 
# #     user_input = input(">")
# #     print("Message Hist")
# #     print(messages)
# #     print("--------")
# #     messages.append(HumanMessage(content="How is weather in Dallas during Febravery? "))
# #     airesponse = Chat(messages)
# #     messages.append(AIMessage(content=airesponse.content))
# #     print(airesponse)
    

# conversation = ConversationChain ( llm=Chat, memory=ConversationBufferMemory(memory_key="history",return_messages=True), verbose=True)

# while True: 
#     user_input = input(">")
#     ai_response = conversation.predict(input = user_input)
#     print(conversation)