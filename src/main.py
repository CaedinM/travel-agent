from resources import open_resources
from agents import build_orchestrator_agent
from state import INITIAL_TRIP_PHASE
from langchain.messages import HumanMessage
import asyncio
import uuid

async def main():
    async with open_resources() as res:

        agent = build_orchestrator_agent()

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        is_new_trip = True

        while True:
            user_input = input("> ")
            if user_input.strip() in {"exit", "quit"}:
                break
            agent_input = {"messages": [HumanMessage(content=user_input)]}
            if is_new_trip:
                agent_input["phase"] = INITIAL_TRIP_PHASE

            result = await agent.ainvoke(agent_input, config=config, context=res)
            is_new_trip = False
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
