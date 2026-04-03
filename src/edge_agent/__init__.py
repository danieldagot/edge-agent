"""edge-agent — A minimal, zero-dependency AI agent framework."""

from edge_agent.agent import Agent, AgentType, Evaluator, Fallback, Guardrail, Router
from edge_agent.chain import Chain
from edge_agent.dotenv import dotenv_values, find_dotenv, load_dotenv
from edge_agent.mcp import MCPServer, load_mcp_config
from edge_agent.schema import parse_dataclass, schema_from_dataclass
from edge_agent.session import Session
from edge_agent.template import render_template
from edge_agent.tool import Tool, tool
from edge_agent.types import (
    AgentStep,
    Message,
    RunResult,
    ToolCall,
    ToolCallRecord,
    ToolResult,
)

__all__ = [
    "Agent",
    "AgentStep",
    "AgentType",
    "Chain",
    "Evaluator",
    "Fallback",
    "Guardrail",
    "MCPServer",
    "load_mcp_config",
    "Message",
    "Router",
    "RunResult",
    "Session",
    "Tool",
    "ToolCall",
    "ToolCallRecord",
    "ToolResult",
    "dotenv_values",
    "find_dotenv",
    "load_dotenv",
    "parse_dataclass",
    "render_template",
    "schema_from_dataclass",
    "tool",
]
