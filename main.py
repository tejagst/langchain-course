from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answere and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources:list[Source] = Field(default_factory=list, description="The list of souces to generate the answer")


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
agent = create_agent(model=llm, tools = tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Search for job postings of ai engineer in the DFW area on LinkedIn with skills around LangChain")})
    print(result)
    



if __name__ == "__main__":
    main()

