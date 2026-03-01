from pydantic import BaseModel, Field

from blackgeorge.tools.base import Tool, ToolResult


class TransferToAgentInput(BaseModel):
    agent_name: str = Field(..., description="The name of the agent to transfer to.")
    context: str = Field(..., description="Context or instructions to pass to the target agent.")


def transfer_to_agent_tool(available_agents: list[str]) -> Tool:
    allowed_agents = tuple(available_agents)
    schema = TransferToAgentInput.model_json_schema()
    properties = schema.get("properties")
    if isinstance(properties, dict):
        agent_name_schema = properties.get("agent_name")
        if isinstance(agent_name_schema, dict):
            agent_name_schema["enum"] = list(allowed_agents)

    def transfer(agent_name: str, context: str) -> ToolResult:
        if agent_name not in allowed_agents:
            available = ", ".join(allowed_agents)
            return ToolResult(
                error=f"Agent '{agent_name}' is not available. Available agents: {available}"
            )
        return ToolResult(
            content=f"Transferred control to {agent_name}.",
            data={"__handoff__": agent_name, "context": context},
        )

    return Tool(
        name="transfer_to_agent",
        description="Transfer control to another agent.",
        schema=schema,
        callable=transfer,
        input_model=TransferToAgentInput,
        requires_handoff=True,
    )
