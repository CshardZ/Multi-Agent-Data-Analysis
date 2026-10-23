from langchain.agents import create_agent
from langchain.tools import tool

from langchain_ollama import ChatOllama


gemma_llm = ChatOllama(
    model='qwen3:4b',
    temperature=0
)


@tool
def square_root(num):
    """Calculate square root of a number"""
    return num ** 0.5


agent = create_agent(
    model=gemma_llm,
    tools=[square_root],
    system_prompt="You are a math calculator."
)


if __name__ == '__main__':
    from langchain.messages import HumanMessage
    print("Agent invoked...")
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is square root of 200?")]}
    )
    print(response)