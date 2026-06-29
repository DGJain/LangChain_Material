# LangChain Material

Hands-on learning material for **LangChain v1**, working through agents, multi-provider
model integration, tools, and the tool execution loop. Built with Python 3.12 and managed
with [`uv`](https://github.com/astral-sh/uv).

## What's inside

The `updatedlangchain/` folder contains a progressive set of Jupyter notebooks plus a
standalone script.

| File | Topic | Status |
|------|-------|--------|
| `1-langchainintro.ipynb` | LangChain v1 intro: building an agent with `create_agent` and a custom tool | ✅ |
| `2-modelintegration.ipynb` | Integrating OpenAI, Google Gemini, and Groq models. Covers `init_chat_model`, `ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatGroq`, plus streaming and batching | ✅ |
| `3-tools.ipynb` | Defining tools with the `@tool` decorator, binding tools to a model, and the tool execution loop | ✅ |
| `4-messages.ipynb` | Messages as the unit of context: text prompts vs. message prompts, the message types (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`), message metadata, and usage metadata | ✅ |
| `5-structuedoutput.ipynb` | Structured output: enforcing a response schema with Pydantic (field validation, descriptions, nested structures), TypedDict, and dataclasses via `with_structured_output`, plus using schemas with `create_agent` | ✅ |
| `6-middleware.ipynb` | Middleware: controlling what happens inside an agent. Covers `SummarizationMiddleware` for compressing long conversation history near token limits, with thread-based checkpointing (`InMemorySaver`) | ✅ |
| `tool_execution_loop.py` | Complete standalone example of the tool execution loop (single-turn + multi-turn agentic loop) using Groq | ✅ |

## Setup

### 1. Prerequisites

- Python >= 3.12
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### 2. Install dependencies

Using `uv`:

```bash
uv sync
```

Or using `pip`:

```bash
pip install -r requirements.txt
```

Core dependencies: `langchain`, `langchain-openai`, `langchain-groq`,
`langchain-google-genai`, `langchain-anthropic`, `langchain-community`,
`python-dotenv`, and `ipykernel` (for running the notebooks).

### 3. Configure API keys

Create a `.env` file in the project root with the keys for the providers you want to use:

```env
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
```

> **Note:** `.env` is git-ignored. Never commit your API keys.

## Running

### Notebooks

Open the `updatedlangchain/` notebooks in Jupyter or VS Code and run the cells top to bottom:

```bash
jupyter notebook updatedlangchain/
```

### Tool execution loop script

```bash
uv run python updatedlangchain/tool_execution_loop.py
```

This script demonstrates the full pattern:

1. Define tools with the `@tool` decorator (`get_weather`, `calculate`, `get_population`)
2. Bind tools to a chat model
3. Invoke the model with a user query
4. Execute the tool calls the model requests
5. Pass results back to the model for a final answer
6. Repeat until no more tool calls (the agentic loop)

It runs two flows: a single-turn loop and a multi-turn agentic loop with a
`MAX_ITERATIONS` safety limit, then prints the full message history.

## Project layout

```
.
├── updatedlangchain/        # notebooks + standalone scripts
│   ├── 1-langchainintro.ipynb
│   ├── 2-modelintegration.ipynb
│   ├── 3-tools.ipynb
│   ├── 4-messages.ipynb
│   ├── 5-structuedoutput.ipynb
│   ├── 6-middleware.ipynb
│   └── tool_execution_loop.py
├── main.py                  # entry-point stub
├── pyproject.toml           # project metadata + dependencies
├── requirements.txt         # pip-installable dependency list
└── uv.lock                  # locked dependency versions
```
