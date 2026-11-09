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

# tools
search_tool = DuckDuckGoSearchRun(max_results=3)

API_KEY = os.getenv("WEATHER_API_KEY") 

@tool
def weather_tool(city: str) -> str:
    """Fetch the current temperature in Celsius for a given city using WeatherAPI."""
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        temp_c = data["current"]["temp_c"]
        return f"The current temperature in {city} is {temp_c}°C."
    else:
        return f"Error: Unable to fetch weather data."
    
@tool
def calculator_tool(first_number: float, second_number:float, operation: str) -> dict:
    """A simple calculator tool to perform basic arithmetic operations.
    Supported operations are add, sub, mul, and div.
    """

    try:
        if operation == "add":
            result = first_number + second_number
        elif operation == "sub":
            result = first_number - second_number
        elif operation == "mul":
            result = first_number * second_number
        elif operation == "div":
            if second_number == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_number / second_number
        else:
            return {"error": f"Unsupported operation: {operation}"}
        
        return {"first_num": first_number, "second_num": second_number, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


tools = [search_tool, weather_tool, calculator_tool]

llm_with_tools = llm.bind_tools(tools)

# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
# graph node

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# checkpointer
conn = sqlite3.connect(database="chatbot_tools.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile()

initial_state = {
    "messages": [
        HumanMessage(content="What's the capital of India? What's the weather there now? How much more or less is it than 20 degree celcius?"),
    ]
}

config = {"configurable": {"thread_id": "thread-3"}}

result = chatbot.invoke(initial_state)

print(result['messages'])