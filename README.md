# edge-agent

**Edge Agent** is a small, modern AI agent framework for Python 3.11+: tools, multi-step tool chaining, MCP servers, typed agent pipelines (guardrails, routers, evaluators, fallbacks), interactive sessions, and a swappable LLM provider — implemented in a **compact, readable codebase** with **zero runtime dependencies** (stdlib only).

Many popular agent stacks make simple workflows feel heavyweight: large dependency trees, lots of boilerplate, and framework surface area that dwarfs the problem you are solving. Edge Agent is built for the opposite: **full-featured agent behavior without the bloat**, so you can ship agents that stay easy to read, test, and own.

> **Built-in providers: Gemini and Ollama.** Edge Agent ships with providers for Google's Gemini models and locally running Ollama instances. The provider interface is open for extension — see [Custom Providers](#custom-providers) to add your own.

## Features

- **Zero runtime dependencies** — uses only the Python standard library
- **Tool support** — define tools with a simple `@tool` decorator
- **Tool chaining** — the agent loop lets the LLM call tools in sequence automatically
- **Per-agent toolsets** — each agent gets its own set of tools, keeping LLM context focused
- **Structured output** — set `output_type` to a dataclass and get typed responses back
- **Template variables** — use `{{currentDate}}`, `{{url:...}}`, or custom placeholders in instructions
- **File-based prompts** — pass a `Path` as instructions to load prompts from disk
- **MCP support** — connect to any MCP server and use its tools natively
- **Chain orchestration** — compose agents into pipelines with built-in control flow
- **Five agent types** — Agent, Guardrail, Router, Evaluator, and Fallback
- **Interactive sessions** — REPL mode with conversation history
- **Provider abstraction** — swap LLM backends without changing agent code
- **Execution tracing** — every `run()` returns a `RunResult` with tool call records, timing, and per-agent steps
- **Built-in logging** — stdlib `logging` under the `edge_agent` namespace

## Installation

```bash
uv add edge-agent
```

Or install from source:

```bash
uv pip install -e .
```

### Package size

Edge Agent has **no runtime dependencies**, so installing it only adds its own source code. There is nothing else to download or resolve.

| Artifact | Size |
|---|---|
| Wheel (`.whl`) — what gets installed | **38 KB** |
| Source distribution (`.tar.gz`) | **41 KB** |

That's **0.038 MB** for the installable wheel — the entire footprint.

A local checkout may show a larger footprint on disk (for example ~200 KB) because of `__pycache__` and compiled `.pyc` files; those are not part of the published wheel. To see the exact distribution size, build the wheel (`uv build`) and check the `.whl` in `dist/`.

## Quick Start

```python
from edge_agent import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = Agent(
    instructions="You research topics and provide summaries.",
    tools=[search],
)

result = agent.run("Research quantum computing")
print(result)          # prints the final text (via __str__)
print(result.output)   # the final text response
print(result.steps)    # execution trace with tool call details
```

`Agent.run()` returns a `RunResult` — see [Execution Tracing](#execution-tracing) for details.

**Note:** Edge Agent ships with **Gemini** and **Ollama** providers. For Gemini, you need a [Google AI API key](https://aistudio.google.com/apikey) — resolved automatically from `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variables (`.env` files are loaded automatically). See [`.env.example`](.env.example) for the required format. For Ollama, see [Ollama provider](#ollama).

## Defining Tools

Use the `@tool` decorator on any typed function. The decorator inspects the function signature and docstring to build the JSON schema automatically:

```python
from edge_agent import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"22 degrees {unit} in {city}"
```

The decorated function remains callable as normal:

```python
get_weather(city="Tokyo")  # "22 degrees celsius in Tokyo"
```

## Tool Chaining

Tool chaining happens automatically. When the LLM calls a tool, the result is fed back into the conversation. The LLM can then call another tool (or the same tool with different arguments) before producing a final text response. A `max_turns` parameter prevents infinite loops:

```python
result = agent.run("Research quantum computing", max_turns=5)
```

## Structured Output

Set `output_type` to a dataclass and the result includes a parsed instance in `result.parsed`. The JSON schema is derived automatically from the dataclass fields, sent to the LLM via Gemini's `responseSchema`, and the JSON response is parsed back into a dataclass instance.

Tools and structured output work together — the agent calls tools as normal, and the final response is parsed into the dataclass:

```python
from dataclasses import dataclass
from edge_agent import Agent, tool

@tool
def get_city_data(city: str) -> str:
    """Look up factual data about a city."""
    return "Tokyo is the capital of Japan. Population: ~14 million."

@dataclass
class CityInfo:
    name: str
    country: str
    population_millions: float
    famous_for: str

agent = Agent(
    instructions="Use the tool to look up facts, then return structured data.",
    tools=[get_city_data],
    output_type=CityInfo,
)

result = agent.run("Tell me about Tokyo.")
print(result.parsed.name)                 # "Tokyo"
print(result.parsed.population_millions)  # 14.0
print(result.output)                      # raw JSON string
```

`output_type` can also be passed per-call via `run(output_type=...)` to override the class-level default:

```python
agent = Agent(instructions="Geography expert.")
result = agent.run("Tell me about Paris.", output_type=CityInfo)
print(result.parsed.name)  # "Paris"
```

Nested dataclasses, `list[...]` fields, and optional fields (with defaults) are all supported.

## Template Variables

Use `{{...}}` placeholders in instructions that get substituted at run time. Pass custom variables via `template_vars`:

```python
agent = Agent(
    instructions=(
        "Today is {{currentDate}}. The user's name is {{userName}}. "
        "Greet them and mention today's date."
    ),
)
result = agent.run("Hi!", template_vars={"userName": "Alice"})
```

Built-in variables:

| Variable | Value |
|---|---|
| `{{currentDate}}` | Today's date in ISO-8601 format (e.g. `2026-03-31`) |
| `{{url:https://...}}` | Fetched URL body (decoded as UTF-8) |

Unknown placeholders are left unchanged, so you can mix built-in and custom variables freely.

## File-Based Prompts

Pass a `pathlib.Path` as instructions to load prompts from a file on disk. The file is read once at agent construction time:

```python
from pathlib import Path
from edge_agent import Agent

agent = Agent(instructions=Path("prompts/assistant.md"))
```

This keeps long prompts out of your Python code and makes them easy to version-control separately.

## MCP (Model Context Protocol)

Connect to any [MCP server](https://modelcontextprotocol.io/) and use its tools as if they were local `@tool` functions. No extra dependencies — Edge Agent implements the MCP client using only `subprocess` and `json` from the standard library.

### Basic usage

```python
from edge_agent import Agent, MCPServer

server = MCPServer(
    "filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

with server:
    print(server.tools)  # tools discovered from the server

    with Agent(
        instructions="You are a helpful file assistant.",
        mcp_servers=[server],
    ) as agent:
        result = agent.run("List the files in /tmp")
        print(result)
```

### How it works

1. `MCPServer.connect()` launches the server as a subprocess
2. Performs the MCP handshake (JSON-RPC 2.0 over stdio)
3. Calls `tools/list` to discover available tools
4. Each tool becomes a regular `Tool` object — the agent loop treats it identically to a `@tool`-decorated function
5. When the LLM calls an MCP tool, the call is proxied to the server via `tools/call`

### MCPServer parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | required | A label for this server (used in logs) |
| `command` | required | The command to launch the server (e.g. `["npx", "-y", "some-mcp-server"]`) |
| `env` | `None` | Extra environment variables to pass to the server process |

### Mixing local and MCP tools

MCP tools are merged with local `@tool` functions. Each tool name must be unique — duplicates raise `ValueError`:

```python
from edge_agent import Agent, MCPServer, tool

@tool
def summarize(text: str) -> str:
    """Summarize text in one sentence."""
    return f"Summary: {text[:80]}..."

server = MCPServer("fs", command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"])

with Agent(
    instructions="Read files and summarize them.",
    tools=[summarize],
    mcp_servers=[server],
) as agent:
    result = agent.run("Read /tmp/notes.txt and summarize it")
```

### Loading servers from a config file

`load_mcp_config()` reads a JSON file that defines multiple MCP servers (the same format used by Claude Desktop) and returns a `dict[str, MCPServer]`. You can load all servers at once or pick specific ones by name:

```python
from edge_agent import Agent, load_mcp_config

# mcp.json
# {
#   "mcpServers": {
#     "filesystem": {
#       "command": "npx",
#       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
#     },
#     "brave-search": {
#       "command": "npx",
#       "args": ["-y", "@modelcontextprotocol/server-brave-search"],
#       "env": { "BRAVE_API_KEY": "sk-..." }
#     }
#   }
# }

# Load every server defined in the file
all_servers = load_mcp_config("mcp.json")

# Load only the servers you need
selected = load_mcp_config("mcp.json", servers=["filesystem"])
fs_server = selected["filesystem"]

with fs_server:
    with Agent(
        instructions="You are a helpful file assistant.",
        mcp_servers=[fs_server],
    ) as agent:
        result = agent.run("List the files in /tmp")
        print(result)
```

`load_mcp_config` parameters:

| Parameter | Default | Description |
|---|---|---|
| `path` | required | Path to the JSON config file (`str` or `pathlib.Path`) |
| `servers` | `None` | List of server names to load; `None` loads all servers |

The function returns unconnected `MCPServer` instances — connection happens when you enter the context manager or call `.connect()`. A `ValueError` is raised if you request a server name that doesn't exist in the config.

### Lifecycle

`MCPServer` supports the context manager protocol. You can also manage the lifecycle manually:

```python
server = MCPServer("my-server", command=["my-mcp-server"])
server.connect()       # launch process + handshake
print(server.tools)    # use tools
server.close()         # terminate process
```

When passed to an `Agent` via `mcp_servers`, the agent auto-connects any servers that aren't already connected. Use `agent.close()` (or the `with Agent(...) as agent:` pattern) to clean up.

## Chain

A `Chain` runs multiple agents sequentially. Each agent in the chain can have its own tools, instructions, and role. The key benefit: **each agent only sends its own tools to the LLM**, keeping the context focused and avoiding the problem of sending every tool on every API call.

```python
from edge_agent import Agent, Chain

chain = Chain(agents=[agent_a, agent_b, agent_c])
result = chain.run("user message")
```

### How Chain works

1. The chain iterates through its agents in order
2. Each agent processes the input and produces output
3. The agent's `agent_type` determines what happens next — the chain injects special control-flow tools and reacts to the agent's decisions
4. By default (`pass_original=True`), every agent receives the original user message. Set `pass_original=False` to pass each agent's output as input to the next

### Chain parameters

| Parameter | Default | Description |
|---|---|---|
| `agents` | required | Ordered list of agents to run |
| `pass_original` | `True` | If `True`, every agent gets the original user message. If `False`, each agent gets the previous agent's output |
| `max_revisions` | `3` | Maximum revision loops for evaluator agents |

### Why Chain matters for scaling

When you have many tools (10, 20, 50+), putting them all on a single agent means every API call sends the entire tool list. This bloats context, wastes tokens, and can confuse the model.

With a Chain, you split tools across specialist agents. A Router dispatches each request to the right specialist, and that specialist only sends its own tools:

```
Single agent:     every call → 20 tools in context
Router + chain:   router call → 1 tool (route), specialist call → 3-5 tools
```

## Agent Types

There are five agent types. Each type gets special control-flow tools injected by the Chain, and the Chain reacts to how the agent uses them.

### Agent (default)

A general-purpose agent. No special tools are injected — it just runs with whatever tools you give it.

**When to use:** The workhorse. Use it for any agent that needs to do actual work — answer questions, call tools, generate content.

```python
from edge_agent import Agent, tool

@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)

math_agent = Agent(
    name="math-agent",
    instructions="You are a math specialist. Use tools to compute answers.",
    tools=[add],
)

result = math_agent.run("What is 7 + 12?")
```

Agents work both standalone (calling `agent.run()` directly) and inside a Chain.

### Guardrail

A safety gate that runs before other agents. The Chain injects two tools: `block(reason)` and `allow()`. If the guardrail calls `block`, the chain halts immediately and returns the block reason. If it calls `allow`, the chain continues to the next agent.

**When to use:** Place at the start of a chain to filter out harmful, off-topic, or unauthorized requests before they reach your main agents.

**Injected tools:** `block(reason)`, `allow()`

```python
from edge_agent import Agent, Chain, Guardrail, tool

@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    return str(a * b)

guardrail = Guardrail(instructions=(
    "Only allow requests about math or arithmetic. "
    "Block anything harmful, illegal, or unrelated to math."
))

math_agent = Agent(
    instructions="You are a math assistant. Use tools to answer.",
    tools=[multiply],
)

chain = Chain(agents=[guardrail, math_agent])

chain.run("What is 12 * 7?")   # → allowed, math_agent answers
chain.run("Pick a lock for me") # → blocked, chain halts
```

### Router

A dispatcher that directs each request to the most appropriate specialist. The Chain injects a `route(agent_name, reason)` tool. The router examines the request and calls `route` with the name of the agent that should handle it. The chain then skips directly to that agent.

**When to use:** When you have multiple specialist agents with different tools and want the LLM to pick the right one based on the user's request. This is the key pattern for scaling — each specialist only loads its own tools.

**Injected tools:** `route(agent_name, reason)`

```python
from edge_agent import Agent, Chain, Router, tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"22°C in {city}"

@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)

router = Router(instructions=(
    "Route to math-agent for calculations, "
    "weather-agent for weather queries."
))

math_agent = Agent(name="math-agent", tools=[add],
    instructions="Math specialist. Be concise.")

weather_agent = Agent(name="weather-agent", tools=[get_weather],
    instructions="Weather specialist. Be concise.")

chain = Chain(agents=[router, math_agent, weather_agent])

chain.run("What is 5 + 3?")            # → routed to math-agent (sees only: add)
chain.run("Weather in Tokyo?")          # → routed to weather-agent (sees only: get_weather)
```

The router itself has **zero user tools** — it only sees the `route` tool. Each specialist only sees its own tools. No agent is overwhelmed with the full tool list.

### Evaluator

A quality reviewer that checks the previous agent's output and can request revisions. The Chain injects `approve()` and `revise(feedback)`. If the evaluator calls `revise`, the chain loops back to the previous agent with the feedback appended, up to `max_revisions` times.

**When to use:** After a content-generating agent when you need quality control. The evaluator-writer loop refines output iteratively — useful for copywriting, code generation, report writing, or anything where a first draft might not be good enough.

**Injected tools:** `approve()`, `revise(feedback)`

```python
from edge_agent import Agent, Chain, Evaluator

writer = Agent(
    name="writer",
    instructions="Write a punchy product tagline. Just the tagline, nothing else.",
)

editor = Evaluator(
    name="editor",
    instructions=(
        "Review the tagline for clarity, impact, and brevity (under 10 words). "
        "Approve if it meets all criteria, otherwise request a revision."
    ),
)

chain = Chain(agents=[writer, editor], max_revisions=2)

chain.run("Wireless noise-cancelling headphones")
# writer drafts → editor reviews → (revise?) → writer redrafts → editor approves
```

### Fallback

An agent that can signal it cannot handle a request, so the chain moves on to the next agent. The Chain injects a `fail(reason)` tool. If the agent handles the request successfully, the chain returns immediately. If it calls `fail`, the chain tries the next agent in line.

**When to use:** When you have specialist agents that should only answer certain types of questions. Stack them in order of specificity, with a generalist at the end as a catch-all.

**Injected tools:** `fail(reason)`

```python
from edge_agent import Agent, Chain, Fallback, tool

@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"22°C in {city}"

math_only = Fallback(
    name="math-only",
    instructions="Answer math questions. For anything else, use fail.",
    tools=[add],
)

weather_only = Fallback(
    name="weather-only",
    instructions="Answer weather questions. For anything else, use fail.",
    tools=[get_weather],
)

generalist = Agent(
    name="generalist",
    instructions="Answer any question concisely.",
)

chain = Chain(agents=[math_only, weather_only, generalist])

chain.run("What is 9 + 4?")           # → math_only handles it
chain.run("Weather in London?")        # → math_only fails → weather_only handles it
chain.run("Who wrote Hamlet?")         # → math_only fails → weather_only fails → generalist handles it
```

## Combining Agent Types

The real power of Edge Agent is composing agent types into pipelines. Each combination solves a different architectural problem. Here are the patterns, when to use them, and how they work step by step.

### Pattern 1: Guardrail + Router + Specialists

**The problem:** You have multiple specialist agents with scoped tools, and you need safety filtering before any of them run.

**The flow:**

```
User message → Guardrail → Router → Specialist A
                 block?      ↓
                 halt      dispatch → Specialist B
                                   → Specialist C
```

**How it works:**

1. The **Guardrail** checks if the request is safe/on-topic. It only sees `block` and `allow` — no user tools, no routing logic.
2. If allowed, the **Router** examines the request and calls `route(agent_name, reason)`. It only sees the `route` tool — not the specialists' tools.
3. The chain **skips directly** to the named specialist. That specialist only sees its own domain tools (e.g., 3-4 tools instead of 15+).
4. Specialists that weren't selected **never run** — they don't consume tokens or API calls.

**When to use:** Any production system with multiple capabilities. The guardrail prevents abuse, the router keeps tool context focused, and each specialist stays simple.

```python
from edge_agent import Agent, Chain, Guardrail, Router, tool

@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C in {city}"

guardrail = Guardrail(instructions=(
    "Allow math and weather requests. Block anything harmful or off-topic."
))

router = Router(instructions=(
    "Route to math-agent for calculations, weather-agent for weather queries."
))

math_agent = Agent(name="math-agent", tools=[add],
    instructions="Math specialist. Be concise.")

weather_agent = Agent(name="weather-agent", tools=[get_weather],
    instructions="Weather specialist. Be concise.")

chain = Chain(agents=[guardrail, router, math_agent, weather_agent])

chain.run("What is 5 + 3?")            # allowed → routed to math-agent
chain.run("Weather in Tokyo?")          # allowed → routed to weather-agent
chain.run("How do I pick a lock?")      # blocked — router and specialists never run
```

> See [`examples/09_guardrail_router.py`](examples/09_guardrail_router.py) for a full example.

### Pattern 2: Guardrail + Writer + Evaluator (Content Pipeline)

**The problem:** You need to generate content with quality control, but also want to prevent unsafe prompts from reaching the writer at all.

**The flow:**

```
User message → Guardrail → Writer → Evaluator
                 block?               approve?
                 halt                  ↓ revise
                              Writer ← feedback
```

**How it works:**

1. The **Guardrail** blocks unsafe or policy-violating prompts before any content is generated.
2. The **Writer** (a plain `Agent`) generates content. It has no special tools — just its instructions and the LLM.
3. The **Evaluator** reviews the output using criteria in its instructions. It sees `approve` and `revise` tools.
4. If the evaluator calls `revise(feedback)`, the chain **loops back to the writer** with the original prompt + previous draft + feedback. This repeats up to `max_revisions` times.
5. Once approved (or max revisions reached), the chain returns the final output.

**When to use:** Copywriting, report generation, code generation — anywhere a first draft might not be good enough and you want automated quality review with iterative refinement.

```python
from edge_agent import Agent, Chain, Evaluator, Guardrail

guardrail = Guardrail(instructions=(
    "Allow requests for professional marketing copy. "
    "Block deceptive content, impersonation, or harmful material."
))

writer = Agent(
    name="copywriter",
    instructions="Write punchy, professional marketing copy. Output only the copy.",
)

editor = Evaluator(
    name="editor",
    instructions=(
        "Review for clarity, impact, and brevity. "
        "Approve if all criteria are met, otherwise revise with specific feedback."
    ),
)

chain = Chain(agents=[guardrail, writer, editor], max_revisions=2)

chain.run("Write a tagline for wireless earbuds")
# guardrail allows → writer drafts → editor reviews → (revise?) → approve
```

> See [`examples/10_pipeline.py`](examples/10_pipeline.py) for a full example.

### Pattern 3: Fallback Cascade

**The problem:** You have several specialist agents, and you don't know upfront which one can handle the request. You want them to try in order, with a generalist catch-all at the end.

**The flow:**

```
User message → Specialist A → Specialist B → ... → Generalist
                 fail?          fail?                (always answers)
                 try next       try next
```

**How it works:**

1. Each `Fallback` agent examines the request. If it's in their domain, they answer normally and the chain **returns immediately**.
2. If it's outside their domain, they call `fail(reason)` and the chain **moves to the next agent**.
3. The last agent is typically a plain `Agent` (no `fail` tool) that catches everything the specialists couldn't handle.

**When to use:** When routing logic is too complex for a single Router (e.g., domain boundaries are fuzzy), or when you want each specialist to self-assess rather than relying on a central dispatcher.

**Router vs. Fallback — how to choose:**

| | Router | Fallback cascade |
|---|---|---|
| **Decision maker** | One router decides for all | Each agent decides for itself |
| **API calls** | 1 (router) + 1 (specialist) = 2 | 1 per agent tried (worst case: all of them) |
| **Best when** | Domains are clearly distinct | Domains overlap or are hard to describe upfront |
| **Scaling** | Cheap — always 2 calls | Expensive if many specialists fail before one handles it |

```python
from edge_agent import Agent, Chain, Fallback, tool

@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"Doc result for: {query}"

@tool
def query_db(sql: str) -> str:
    """Run a database query."""
    return f"DB result for: {sql}"

docs_agent = Fallback(
    name="docs-agent",
    instructions="Answer documentation questions. Fail for anything else.",
    tools=[search_docs],
)

db_agent = Fallback(
    name="db-agent",
    instructions="Answer database questions. Fail for anything else.",
    tools=[query_db],
)

generalist = Agent(
    name="generalist",
    instructions="Answer any question as best you can.",
)

chain = Chain(agents=[docs_agent, db_agent, generalist])
```

### Pattern 4: Full Pipeline (All Types Combined)

**The problem:** You need everything — safety, intelligent routing, specialist tools, graceful degradation, and quality review — in one pipeline.

**The flow:**

```
User message → Guardrail → Router → Specialist (Fallback) → Evaluator
                 block?      ↓          fail?                  revise?
                 halt      dispatch     try next              loop back
```

**How it works:**

1. **Guardrail** blocks unsafe requests at the gate — nothing else runs.
2. **Router** dispatches to the right specialist based on the request content.
3. **Specialists** (as `Fallback` agents) attempt to handle the request. If one can't, the chain tries the next until one succeeds or the chain reaches a generalist.
4. **Evaluator** at the end reviews whatever output was produced and can request revisions.

**When to use:** Complex production systems where you need every layer of control. This is the most expensive pattern (more API calls), so use it when quality and safety are critical.

```python
from edge_agent import Agent, Chain, Evaluator, Fallback, Guardrail, Router, tool

@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"22°C in {city}"

@tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Doc: {query}"

guardrail = Guardrail(
    name="safety",
    instructions="Allow math, weather, and support questions. Block harmful requests.",
)

router = Router(
    name="dispatcher",
    instructions=(
        "Route to: math-specialist for calculations, "
        "weather-specialist for weather, support for product questions."
    ),
)

math_specialist = Fallback(
    name="math-specialist",
    instructions="Handle math questions. Fail for anything else.",
    tools=[add],
)

weather_specialist = Fallback(
    name="weather-specialist",
    instructions="Handle weather questions. Fail for anything else.",
    tools=[get_weather],
)

support = Agent(
    name="support",
    instructions="Answer product and policy questions.",
    tools=[search_docs],
)

quality = Evaluator(
    name="quality",
    instructions="Review for accuracy and clarity. Approve or revise.",
)

chain = Chain(
    agents=[guardrail, router, math_specialist, weather_specialist, support, quality],
    max_revisions=1,
)
```

> See [`examples/11_full_chain.py`](examples/11_full_chain.py) for a full example.

### Choosing the Right Pattern

| Pattern | Agents used | API calls per request | Best for |
|---|---|---|---|
| **Single agent** | Agent | 1+ (tool chaining) | Simple tasks, few tools |
| **Guardrail + Agent** | Guardrail → Agent | 2+ | Safety-gated single capability |
| **Router + Specialists** | Router → Agent(s) | 2+ | Multiple capabilities, clear domains |
| **Guardrail + Router** | Guardrail → Router → Agent(s) | 3+ | Safety + multiple capabilities |
| **Writer + Evaluator** | Agent → Evaluator | 2+ (with revision loops) | Content generation with quality control |
| **Content Pipeline** | Guardrail → Agent → Evaluator | 3+ | Safety + generation + quality |
| **Fallback Cascade** | Fallback(s) → Agent | 1-N | Fuzzy domains, self-assessing specialists |
| **Full Pipeline** | All types | 4+ | Production systems needing every layer |

## Agent Type Reference

| Type | Class | Injected tools | Chain behavior |
|---|---|---|---|
| `"agent"` | `Agent` | none | Runs normally |
| `"guardrail"` | `Guardrail` | `block(reason)`, `allow()` | `block` halts the chain |
| `"router"` | `Router` | `route(agent_name, reason)` | Runs the named agent, returns its result |
| `"evaluator"` | `Evaluator` | `approve()`, `revise(feedback)` | `revise` loops back to previous agent |
| `"fallback"` | `Fallback` | `fail(reason)` | `fail` skips to the next agent |

## Execution Tracing

Every call to `Agent.run()` or `Chain.run()` returns a `RunResult` with full execution traces — what tools were called, with what arguments, what they returned, and how long each took.

### RunResult structure

| Field | Type | Description |
|---|---|---|
| `output` | `str` | The final text response |
| `parsed` | `Any \| None` | The parsed dataclass when `output_type` is used |
| `steps` | `list[AgentStep]` | One entry per agent that ran |

`str(result)` returns `result.output`, so `print(result)` and f-strings work naturally.

### AgentStep fields

| Field | Type | Description |
|---|---|---|
| `agent_name` | `str` | Name of the agent |
| `agent_type` | `str` | `"agent"`, `"guardrail"`, `"router"`, etc. |
| `tools_used` | `list[ToolCallRecord]` | Every tool call made during this step |
| `output` | `str` | The agent's text output |
| `turns` | `int` | Number of LLM turns used |

### ToolCallRecord fields

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Tool name |
| `arguments` | `dict` | Arguments passed to the tool |
| `result` | `str` | The tool's return value |
| `duration_ms` | `float` | Execution time in milliseconds |

### Single agent example

```python
from edge_agent import Agent, tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = Agent(instructions="You are a calculator.", tools=[calculate])
result = agent.run("What is 2 + 2?")

print(result.output)                     # "The answer is 4."
print(result.steps[0].agent_name)        # "agent-1"
print(result.steps[0].tools_used[0].name)        # "calculate"
print(result.steps[0].tools_used[0].arguments)   # {"expression": "2 + 2"}
print(result.steps[0].tools_used[0].result)      # "4"
print(result.steps[0].tools_used[0].duration_ms) # 0.02
```

### Chain example

With a Chain, `result.steps` contains one entry per agent that ran, so you can trace the full pipeline:

```python
from edge_agent import Agent, Chain, Guardrail, Evaluator

guard = Guardrail(name="safety", instructions="Allow safe requests.")
writer = Agent(name="writer", instructions="Write copy.")
reviewer = Evaluator(name="reviewer", instructions="Review quality.")

chain = Chain(agents=[guard, writer, reviewer])
result = chain.run("Write a tagline for headphones")

for step in result.steps:
    print(f"{step.agent_name} ({step.agent_type}): {len(step.tools_used)} tool(s)")
# safety (guardrail): 1 tool(s)    — called allow()
# writer (agent): 0 tool(s)        — produced the tagline
# reviewer (evaluator): 1 tool(s)  — called approve()
```

## Session

Wrap any agent in a `Session` for an interactive terminal REPL with conversation history preserved across turns:

```python
from edge_agent import Agent, Session, tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = Agent(instructions="You are a helpful assistant.", tools=[search])
session = Session(agent)
session.start()
```

## Providers

### Gemini (default)

```python
from edge_agent.providers import GeminiProvider

provider = GeminiProvider(
    model="gemini-3.1-flash-lite-preview",  # optional, has a sensible default
    api_key="your-key",                     # optional, resolved from env
    verify_ssl=False,                       # optional, default True — disable for local dev
)
```

You can also set the default model with the **`EDGE_AGENT_MODEL`** environment variable (legacy: **`TINYAGENT_MODEL`**).

TLS certificate verification is **enabled by default**. To disable it (e.g. behind a corporate proxy), pass `verify_ssl=False` or set `EDGE_AGENT_VERIFY_SSL=false` in your environment.

### Ollama

Run agents against models on your local machine via [Ollama](https://ollama.com/). No API key required.

```python
from edge_agent import Agent
from edge_agent.providers import OllamaProvider

provider = OllamaProvider(
    model="llama3.2",                         # optional, default "llama3.2"
    base_url="http://localhost:11434",         # optional, default localhost
    timeout=120,                              # optional, default 120s
)

agent = Agent(instructions="You are a helpful assistant.", provider=provider)
result = agent.run("Hello!")
print(result)
```

| Parameter | Default | Env var | Description |
|---|---|---|---|
| `model` | `"llama3.2"` | `OLLAMA_MODEL` | Ollama model name (e.g. `mistral`, `codellama`) |
| `base_url` | `"http://localhost:11434"` | `OLLAMA_HOST` | Ollama server URL |
| `timeout` | `120` | — | Request timeout in seconds |

All agent types (`Agent`, `Guardrail`, `Router`, `Evaluator`, `Fallback`), `Chain`, `Session`, structured output (`output_type`), and tool calling work with the Ollama provider. Use a model that supports function calling and structured output (e.g. `llama3.2`, `mistral`).

See [`examples/14_ollama.py`](examples/14_ollama.py) for a runnable local demo using `OllamaProvider` with tools.

### Custom Providers

You can add support for any LLM by implementing the `Provider` abstract class:

```python
from edge_agent.providers import Provider
from edge_agent import Message, Tool

class MyProvider(Provider):
    def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        output_schema: dict[str, object] | None = None,
    ) -> Message:
        # Make your API call and return a Message
        # output_schema is a JSON schema dict for structured output
        ...
```

## Logging

Logging is silent by default. Opt in by configuring the `edge_agent` logger:

```python
import logging

logging.basicConfig()
logging.getLogger("edge_agent").setLevel(logging.DEBUG)
```

## Examples

See the [`examples/`](examples/) directory:

| Example | What it demonstrates |
|---|---|
| `01_hello.py` | Minimal agent, no tools |
| `02_tools.py` | `@tool` decorator, multi-step tool chaining |
| `03_session.py` | Interactive REPL with conversation history |
| `04_guardrail.py` | Guardrail → worker chain |
| `05_router.py` | Router → specialist dispatch |
| `06_evaluator.py` | Writer → evaluator revision loop |
| `07_fallback.py` | Fallback cascade with generalist catch-all |
| `08_custom_provider.py` | Custom LLM provider implementation |
| `09_guardrail_router.py` | Guardrail + Router + specialists combined |
| `10_pipeline.py` | Content pipeline: Guardrail → Writer → Evaluator |
| `11_full_chain.py` | All agent types in one chain |
| `12_mcp.py` | MCP server connection and tool usage |
| `13_mcp_config.py` | Load MCP servers from a JSON config file |
| `14_ollama.py` | Explicit `OllamaProvider` demo with local tool calling |
| `multi_tool_demo/` | Tools across multiple files, flat vs. router comparison |
| `mcp_demo/` | Load MCP servers from a `mcp.json` config file and run an agent against them |
| [`advanced_features_demo.py`](advanced_features_demo.py) | Template variables, file-based prompts, and structured output |

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest tests/ -v
```
