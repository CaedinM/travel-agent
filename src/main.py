from resources import open_resources
from agents import build_orchestrator_agent
from langchain.messages import HumanMessage
import asyncio
import uuid

async def main():
    async with open_resources() as res:

        agent = build_orchestrator_agent()

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        while True:
            user_input = input("> ")
            if user_input.strip() in {"exit", "quit"}:
                break
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                context=res
                )
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())