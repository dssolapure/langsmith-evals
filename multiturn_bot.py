from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
import requests
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from IPython.display import Image

#environment variables
load_dotenv()
llm = ChatOpenAI(model_name = 'gpt-4o')

# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
# graph node

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# initial state
initial_state = {
    "messages": [
        HumanMessage(content="What about the dish 2? How long does it take to prepare it?")
    ]
}

config = {"configurable": {"thread_id": "thread-001"}}

# output
result = chatbot.invoke(initial_state, config=config)

print(result['messages'])