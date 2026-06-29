"""
Tool Execution Loop - Complete Example
========================================
This script demonstrates the full tool execution loop pattern in LangChain:
  1. Define tools with @tool decorator
  2. Bind tools to a chat model
  3. Invoke the model with a user query
  4. Check if the model requested tool calls
  5. Execute each tool call and collect results
  6. Pass results back to the model for the final response
  7. Repeat until no more tool calls (agentic loop)
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# ─── Load environment variables ───────────────────────────────────────────────
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ─── Step 1: Define Tools ────────────────────────────────────────────────────
# Each tool needs a docstring — it tells the LLM when and how to use the tool.

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # Simulated weather data
    weather_data = {
        "new york": "72°F and partly cloudy",
        "london": "15°C and rainy",
        "tokyo": "28°C and humid",
        "paris": "20°C and sunny",
        "mumbai": "34°C and hot",
    }
    return weather_data.get(location.lower(), f"Weather data not available for {location}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result. Example: '2 + 3 * 4'"""
    try:
        result = eval(expression)  # In production, use a safe math parser
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_population(city: str) -> str:
    """Get the approximate population of a city."""
    population_data = {
        "new york": "8.3 million",
        "london": "8.9 million",
        "tokyo": "13.9 million",
        "paris": "2.1 million",
        "mumbai": "20.7 million",
    }
    return population_data.get(city.lower(), f"Population data not available for {city}")


# Collect all tools into a list and build a lookup dictionary
tools = [get_weather, calculate, get_population]
tool_map = {t.name: t for t in tools}  # {"get_weather": get_weather, ...}

# ─── Step 2: Initialize model and bind tools ────────────────────────────────
model = init_chat_model("groq:qwen/qwen3-32b")
model_with_tools = model.bind_tools(tools)

# ─── Step 3: Single-Turn Tool Execution Loop ────────────────────────────────
# This is the simplest form — one user message, one round of tool calls.

print("=" * 60)
print("SINGLE-TURN TOOL EXECUTION LOOP")
print("=" * 60)

# User asks a question that requires a tool
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

# Model decides whether to call a tool
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Execute each tool the model requested
for tool_call in ai_msg.tool_calls:
    print(f"  [TOOL] Called : {tool_call['name']}")
    print(f"         Args   : {tool_call['args']}")

    # Look up the tool by name and invoke it with the tool_call dict
    selected_tool = tool_map[tool_call["name"]]
    tool_result = selected_tool.invoke(tool_call)
    messages.append(tool_result)

    print(f"Result : {tool_result.content}")

# Pass everything back to the model for the final answer
final_response = model_with_tools.invoke(messages)
print(f"\n  [ANSWER] {final_response.text}")


# ─── Step 4: Multi-Turn Agentic Loop ────────────────────────────────────────
# The model may need multiple rounds of tool calls before it can answer.
# This loop keeps going until the model responds with text (no more tool calls).

print("\n" + "=" * 60)
print("MULTI-TURN AGENTIC LOOP")
print("=" * 60)

# A complex question that may require multiple tools
messages = [
    {
        "role": "user",
        "content": "What's the weather in Mumbai and what is its population? "
                   "Also calculate 1500 * 12.",
    }
]

MAX_ITERATIONS = 10  # Safety limit to prevent infinite loops

for i in range(MAX_ITERATIONS):
    print(f"\n--- Iteration {i + 1} ---")

    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    # If no tool calls, the model is done — it has the final answer
    if not ai_msg.tool_calls:
        print(f"\n  [ANSWER]\n{ai_msg.text}")
        break

    # Execute every tool call the model made in this iteration
    for tool_call in ai_msg.tool_calls:
        print(f"  [TOOL] Called : {tool_call['name']}")
        print(f"         Args   : {tool_call['args']}")

        selected_tool = tool_map[tool_call["name"]]
        tool_result = selected_tool.invoke(tool_call)
        messages.append(tool_result)

        print(f"         Result : {tool_result.content}")
else:
    print("[WARNING] Reached maximum iterations without a final answer.")


# ─── Step 5: Inspect the full message history ───────────────────────────────
print("\n" + "=" * 60)
print("FULL MESSAGE HISTORY")
print("=" * 60)

for idx, msg in enumerate(messages):
    if isinstance(msg, dict):
        print(f"  [{idx}] {msg['role'].upper()}: {msg['content'][:80]}")
    else:
        role = msg.__class__.__name__
        content = (msg.content or "(tool call request)")[:80]
        print(f"  [{idx}] {role}: {content}")
