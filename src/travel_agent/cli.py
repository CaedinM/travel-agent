import asyncio
import uuid

from langchain.messages import HumanMessage

from travel_agent.agents.orchestrator import build_orchestrator_agent
from travel_agent.core.state import INITIAL_TRIP_PHASE
from travel_agent.core.responses import PlannerResponse
from travel_agent.infrastructure.resources import open_resources


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
            structured_response = result.get("structured_response")
            if isinstance(structured_response, PlannerResponse):
                structured_response = structured_response.for_trip_state(result)
                print(structured_response.model_dump_json(indent=2))
            else:
                print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
