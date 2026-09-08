from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents import create_agent


model = init_chat_model(model="gemma4", model_provider="openai", api_key="dummy", base_url="http://localhost:8080/v1")
# model = init_chat_model(model="gemini-3.1-flash-lite", model_provider="google-genai")
agent = create_agent(model=model)

question = HumanMessage(content="Who is the chief minister of Tamil Nadu?")

response = agent.invoke(
    {"messages": [question]}
)

print(response['messages'][-1].content)

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for a single, isolated piece of information.
    
    Args:
        query: A concise, atomic search phrase focused on ONE specific entity, 
               person, or metric. NEVER combine multiple questions, use conjunctions 
               like 'and'/'or', or pass full sentences.
    """
    return tavily_client.search(query, max_results=1, include_answer=True, search_depth='fast')

web_search.invoke("Who is the current CM of tamil nadu?")

from datetime import datetime
from langchain_core.tools import tool

@tool
def get_current_date() -> str:
    """Returns today's current date and year. 
    
    CRITICAL INSTRUCTION: You MUST call this tool immediately whenever the user asks 
    about current events, things happening "today", "now", "this year", or whenever 
    up-to-date time context is required to ground your search queries or response.
    """
    return datetime.now().strftime("%Y-%m-%d")

agent = create_agent(
    model=model,
    tools=[web_search, get_current_date]
)

question = HumanMessage(content="What are the latest tragedy events that happened in the following locations: 1.Kerala, 2.Nepal, 3.Delhi and 4.Maharashtra?")

response = agent.invoke(
    {"messages": [question]}
)
from pprint import pprint

pprint(response['messages'])


