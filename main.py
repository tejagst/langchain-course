from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


'''
#### Custom tool for Tavily searching a query with Tavily API Call

from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Return:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query = query)
'''

llm = ChatOpenAI()
tools = [TavilySearch()]  ####[search] in case of custom tool
agent = create_agent(model=llm, tools = tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokyo")})
    print(result)
    



if __name__ == "__main__":
    main()

