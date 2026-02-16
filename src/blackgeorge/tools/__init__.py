from typing import Any

from blackgeorge.tools.base import Tool, ToolPostHook, ToolPreHook, ToolResult
from blackgeorge.tools.decorators import tool
from blackgeorge.tools.execution import execute_tool
from blackgeorge.tools.image_generation import agenerate_image, generate_image
from blackgeorge.tools.mcp import MCPToolProvider
from blackgeorge.tools.registry import Toolbelt

Toolkit = Toolbelt


def create_subworker_tool(*args: Any, **kwargs: Any) -> Tool:
    from blackgeorge.tools.swarm import create_subworker_tool as _create_subworker_tool

    return _create_subworker_tool(*args, **kwargs)


def create_swarm_tool(*args: Any, **kwargs: Any) -> Tool:
    from blackgeorge.tools.swarm import create_swarm_tool as _create_swarm_tool

    return _create_swarm_tool(*args, **kwargs)


__all__ = [
    "MCPToolProvider",
    "Tool",
    "ToolPostHook",
    "ToolPreHook",
    "ToolResult",
    "Toolbelt",
    "Toolkit",
    "agenerate_image",
    "create_subworker_tool",
    "create_swarm_tool",
    "execute_tool",
    "generate_image",
    "tool",
]
