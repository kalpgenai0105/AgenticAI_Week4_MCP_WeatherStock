"""
agent.py
========
A LangGraph ReAct agent that answers natural-language questions by calling
tools hosted on weather_stock_server.py via the Model Context Protocol (MCP).

HOW IT FITS INTO THE SYSTEM
─────────────────────────────
  python agent.py
       │
       ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  agent.py  (this file)                                         │
  │                                                                │
  │  1. build_llm()        → creates the LLM (Gemini by default)  │
  │                                                                │
  │  2. MultiServerMCPClient                                       │
  │     └─ spawns weather_stock_server.py as a subprocess         │
  │     └─ communicates over stdin/stdout (stdio transport)        │
  │     └─ fetches the list of available tools via MCP protocol   │
  │                                                                │
  │  3. create_react_agent(llm, tools)                             │
  │     └─ builds a LangGraph compiled state machine              │
  │     └─ implements the ReAct loop:                              │
  │          Reason → pick a tool → Act (call it) → Reason → …   │
  │                                                                │
  │  4. agent.ainvoke(query)                                       │
  │     └─ runs the loop for each demo query                      │
  │     └─ prints the final answer                                 │
  └────────────────────────────────────────────────────────────────┘
       │  JSON-RPC over stdio
       ▼
  weather_stock_server.py  (MCP server subprocess)
       │
       ├─► OpenWeather REST API  (get_current_weather tool)
       └─► Yahoo Finance         (get_stock_data tool)

WHAT IS ReAct?
──────────────
ReAct (Reason + Act) is a prompting / agent-loop strategy where the LLM
alternates between:
  • Reasoning  – "I need the weather for London; I'll call get_current_weather."
  • Acting      – calls the tool and receives a result
  • Reasoning  – "The result shows 18 °C. I can now answer the user."
  • Final answer – returns text to the user

LangGraph's create_react_agent implements this as a compiled state graph
with two nodes: an LLM node and a tool-execution node.  The graph loops
until the LLM produces a message with no tool calls.

WHAT IS MCP?
────────────
The Model Context Protocol (MCP) is an open standard (Anthropic, 2024) for
connecting LLM agents to external data sources and tools via a well-defined
JSON-RPC interface.  The key advantage over plain Python functions is that
tools are *server processes*: they can be written in any language, restarted
independently, and shared across multiple agents.

langchain-mcp-adapters bridges MCP and LangChain: it fetches the tool schemas
from the server and wraps each one as a standard LangChain BaseTool that
create_react_agent can call transparently.

USAGE
─────
    python agent.py

REQUIRED ENVIRONMENT VARIABLES
───────────────────────────────
    GEMINI_API_KEY        – Google Gemini API key
                            (also accepted as GOOGLE_API_KEY)
    OPENWEATHER_API_KEY   – Passed to the MCP server subprocess so it can
                            call the OpenWeather API

OPTIONAL ENVIRONMENT VARIABLES
───────────────────────────────
    GEMINI_MODEL          – Gemini model name (default: gemini-3.8-flash)
    LLM_PROVIDER          – "gemini" (default) | "anthropic" | "openai"
                            Edit build_llm() below to add more providers.
"""

import asyncio          # Standard library async event loop
import os               # Environment variable access
import sys              # sys.executable (current Python path) and sys.exit()
from pathlib import Path  # Cross-platform path building

# MultiServerMCPClient: starts MCP server subprocesses, speaks the MCP
# JSON-RPC protocol, and exposes each server's tools as LangChain BaseTools.
from langchain_mcp_adapters.client import MultiServerMCPClient

# create_react_agent: LangGraph factory that builds a compiled ReAct graph.
# It accepts any LangChain ChatModel and a list of LangChain-compatible tools.
from langgraph.prebuilt import create_react_agent


# ═══════════════════════════════════════════════════════════════════════════════
# LLM FACTORY  (provider-configurable)
# ═══════════════════════════════════════════════════════════════════════════════
#
# All three providers (Gemini, Anthropic, OpenAI) expose the same
# langchain_core.language_models.chat_models.BaseChatModel interface, which is
# what create_react_agent expects.  Isolating the provider choice in one function
# means swapping LLMs is a one-line environment variable change with no other
# code edits required.
#
def build_llm():
    """Instantiate and return the configured LLM chat model.

    Reads LLM_PROVIDER from the environment (default: "gemini") and returns
    the corresponding LangChain ChatModel instance.  Calls sys.exit() with a
    clear message if the required API key is missing — better than an opaque
    exception later in the async loop.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

    # ── Google Gemini (default) ────────────────────────────────────────────────
    #
    # langchain-google-genai wraps the Google Generative AI Python SDK.
    # It reads the key from the GOOGLE_API_KEY env var, but we also accept the
    # friendlier alias GEMINI_API_KEY and normalise it before constructing the
    # client so that the underlying library finds it under the name it expects.
    #
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            sys.exit(
                "ERROR: Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running."
            )
        # setdefault writes GOOGLE_API_KEY only if it is not already set, so
        # if the user already has GOOGLE_API_KEY this is a no-op.
        os.environ.setdefault("GOOGLE_API_KEY", api_key)

        # Deferred import: we only import the provider library that is actually
        # used.  This avoids an ImportError at startup when an alternative
        # provider is selected but its package is not installed.
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        # temperature=0 → deterministic output, which is better for tool-calling
        # agents because the LLM should reliably pick the right tool rather than
        # exploring creatively.
        #
        # thinking_budget=0 disables Gemini's "thinking" mode.  Thinking models
        # attach a thought_signature to every tool call, but LangChain does not
        # forward those signatures on the follow-up turn, which causes a 400 error.
        # Disabling thinking avoids that incompatibility entirely.
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            thinking_budget=0,
        )

    # ── Anthropic Claude ───────────────────────────────────────────────────────
    #
    # Set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY to use Claude.
    # Install: pip install langchain-anthropic
    #
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ERROR: Set ANTHROPIC_API_KEY before running.")
        from langchain_anthropic import ChatAnthropic  # type: ignore

        # claude-haiku is the fastest / cheapest Claude model — good for demos.
        # Swap to "claude-sonnet-4-5" or "claude-opus-4-5" for harder tasks.
        return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    #
    # Set LLM_PROVIDER=openai and OPENAI_API_KEY to use GPT.
    # Install: pip install langchain-openai
    #
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("ERROR: Set OPENAI_API_KEY before running.")
        from langchain_openai import ChatOpenAI  # type: ignore

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ── Unknown provider ───────────────────────────────────────────────────────
    sys.exit(
        f"ERROR: Unknown LLM_PROVIDER '{provider}'. "
        "Choose 'gemini', 'anthropic', or 'openai'."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO QUERIES
# ═══════════════════════════════════════════════════════════════════════════════
#
# Three queries that exercise progressively more of the system:
#   Query 1 – single tool (weather only)
#   Query 2 – single tool (stock only)
#   Query 3 – multi-tool (the agent must call *both* tools and synthesise
#              the results into one coherent answer)
#
# These also serve as a quick smoke test: if queries 1 and 2 pass but query 3
# fails, the problem is likely in the agent's reasoning, not the tools.
#
DEMO_QUERIES = [
    "What is the current weather in London? Include temperature and humidity.",
    "What is the current stock price of Apple (AAPL)?",
    (
        "Briefly compare today's weather in New York with Tesla (TSLA) "
        "stock performance over the last month. Is the weather or the stock "
        "more volatile right now?"
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT COROUTINE
# ═══════════════════════════════════════════════════════════════════════════════

async def run_agent() -> None:
    """Build the LLM, connect to the MCP server, and run each demo query."""

    # ── Step 1: Build the LLM ─────────────────────────────────────────────────
    llm = build_llm()

    # ── Step 2: Resolve the MCP server script path ────────────────────────────
    #
    # Using Path(__file__).parent ensures the path is correct regardless of
    # which directory the user runs `python agent.py` from.
    # str() converts it to a plain string because subprocess.Popen (used
    # internally by the MCP client) expects a str, not a Path object.
    #
    server_script = str(Path(__file__).parent / "weather_stock_server.py")

    # ── Step 3: Define the MCP server connection config ───────────────────────
    #
    # MultiServerMCPClient accepts a dict keyed by an arbitrary server name.
    # Each value describes how to connect to one MCP server.
    #
    # "transport": "stdio"
    #   The client will spawn "command args…" as a subprocess and communicate
    #   with it over that process's stdin and stdout pipes using the MCP
    #   JSON-RPC framing.
    #
    # "command": sys.executable
    #   sys.executable is the absolute path to the Python interpreter currently
    #   running this script.  Using it (instead of a bare "python" or "python3")
    #   guarantees the server runs inside the same virtual environment, which
    #   means it has access to the same installed packages (mcp, yfinance, etc.).
    #
    # Environment inheritance:
    #   The subprocess inherits the full environment of the parent process,
    #   including OPENWEATHER_API_KEY, which the server reads at tool-call time.
    #   No explicit env= argument is needed.
    #
    server_config = {
        "weather_stock": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [server_script],
        }
    }

    print("Starting MCP server …", flush=True)
    # flush=True is important here: when stdout is connected to a pipe (e.g.
    # in CI or when output is redirected), Python buffers writes by default and
    # the message might not appear until much later.  flush forces it out now.

    # ── Step 4: Connect to the MCP server ─────────────────────────────────────
    #
    # MultiServerMCPClient is used as an async context manager.
    #
    # On __aenter__:
    #   • Spawns the subprocess (weather_stock_server.py)
    #   • Sends the MCP "initialize" handshake
    #   • Sends "tools/list" and downloads the schema for every registered tool
    #
    # On __aexit__:
    #   • Sends the MCP "shutdown" message
    #   • Closes the stdin/stdout pipes, which causes the subprocess to exit
    #
    # Everything inside the `async with` block can use the tools.
    # Outside the block, the server process is gone.
    #
    # As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient no longer supports
    # use as a context manager.  Create the client directly and await get_tools().
    mcp_client = MultiServerMCPClient(server_config)

    # ── Step 5: Retrieve tools as LangChain BaseTools ─────────────────────
    #
    # get_tools() returns a list of langchain_core.tools.BaseTool instances,
    # one per tool the server advertised.  Each BaseTool has:
    #   .name        – the function name (e.g. "get_current_weather")
    #   .description – the tool's docstring (used by the LLM to decide
    #                  *when* to call the tool)
    #   .args_schema – a Pydantic model derived from the tool's JSON Schema
    #                  (used by the LLM to know *what arguments* to pass)
    #
    # create_react_agent receives these BaseTools directly; it never needs
    # to know that they are MCP-backed rather than plain Python functions.
    #
    tools = await mcp_client.get_tools()
    print(f"Loaded {len(tools)} tool(s): {[t.name for t in tools]}\n", flush=True)

    # ── Step 6: Build the ReAct agent ─────────────────────────────────────
    #
    # create_react_agent returns a LangGraph CompiledGraph — a state machine
    # that implements the ReAct loop.  The graph has two nodes:
    #
    #   ┌─────────────┐         ┌────────────────┐
    #   │  LLM node   │──calls──►  Tool node      │
    #   │  (reason)   │◄──result─┤  (act/execute) │
    #   └──────┬──────┘         └────────────────┘
    #          │ no more tool calls
    #          ▼
    #      final answer
    #
    # The graph runs until the LLM produces a message with no tool_calls,
    # at which point it returns the full message history.
    #
    # model=llm   – the chat model to use for reasoning
    # tools=tools – the list of tools the LLM is allowed to call
    #
    # Note: we do NOT pass a checkpointer here, so the agent has no memory
    # across separate ainvoke() calls.  Each query starts fresh.
    # Add `checkpointer=MemorySaver()` + a `thread_id` in the config dict
    # below if you want conversation memory across turns.
    #
    agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    # ── Step 7: Run each demo query ────────────────────────────────────────
    for idx, query in enumerate(DEMO_QUERIES, start=1):
        separator = "=" * 68
        print(f"\n{separator}")
        print(f"Query {idx}/{len(DEMO_QUERIES)}: {query}")
        print(separator)

        # agent.ainvoke() is the async version of agent.invoke().
        # We use the async variant because we are already inside an async
        # function (run_agent), and using the sync variant here would block
        # the event loop and prevent the MCP client's background tasks from
        # running, which would deadlock the stdio communication.
        #
        # Input format:
        #   {"messages": [...]}  – a LangGraph state dict.
        #   Each message is a dict with "role" and "content" keys,
        #   following the OpenAI chat format that LangChain normalises.
        #
        # config:
        #   recursion_limit=20  – maximum number of graph steps (LLM calls +
        #                         tool calls combined) before LangGraph raises
        #                         a GraphRecursionError.  Prevents runaway
        #                         loops if the LLM keeps calling tools forever.
        #                         20 is generous for two simple tools; reduce
        #                         to 10 for tighter cost control.
        #
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 20},
        )

        # ── Step 8: Extract and print the final answer ─────────────────────
        #
        # result["messages"] is the full conversation history as a list of
        # LangChain BaseMessage objects:
        #   [HumanMessage, AIMessage(tool_calls), ToolMessage, ..., AIMessage]
        #
        # The final element is always the last AIMessage — the assistant's
        # answer to the user after all tool calls are resolved.
        # We access .content to get the plain text string.
        #
        answer = result["messages"][-1].content
        print(f"\nAnswer:\n{answer}")

    print("\nMCP server shut down. Done.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
#
# asyncio.run() creates a new event loop, runs the coroutine to completion,
# and then closes the loop.  It is the standard way to launch a top-level
# async function from synchronous code.
#
# Why async at all?
#   MultiServerMCPClient uses asyncio under the hood for non-blocking subprocess
#   I/O (reading from the server's stdout without blocking the Python thread).
#   The LangGraph agent's .ainvoke() also uses asyncio for concurrent tool calls.
#   Making run_agent() async lets both run on the same event loop without threads.
#
if __name__ == "__main__":
    asyncio.run(run_agent())
