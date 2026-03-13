"""
marco-code - A minimal LLM-powered coding assistant.

This implementation demonstrates the core "agent loop" pattern used by AI coding
assistants. The LLM iteratively calls tools until it has enough information to
respond to the user.

Agent loop data flow:

    User prompt (-p)
           │
           ▼
    ┌─────────────┐
    │  messages[]  │◄──────────────────────────────┐
    │  (history)   │                               │
    └──────┬──────┘                               │
           │                                       │
           ▼                                       │
    ┌─────────────┐    finish_reason="stop"       │
    │ call_model() ├──────────────────────► print() & exit
    │  (OpenRouter)│                               │
    └──────┬──────┘                               │
           │ tool_calls?                           │
           ▼                                       │
    ┌─────────────┐    result string              │
    │ TOOLS[name] ├───────────────────────────────┘
    │ read/write/ │
    │ bash        │
    └─────────────┘
"""

import argparse
import os
import platform
import sys
import json
import subprocess

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration for OpenRouter API (OpenAI-compatible endpoint)
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

# Module-level client singleton (lazy-initialized on first call)
_client = None

MAX_ITERATIONS = 10
BASH_TIMEOUT = 30  # seconds


def _get_client():
    """Return the shared OpenAI client, creating it on first use."""
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def call_model(messages, tools):
    """
    Send a request to the LLM with conversation history and available tools.

    Args:
        messages: List of conversation messages (user, assistant, tool results)
        tools: List of tool definitions the LLM can choose to call

    Returns:
        The API response containing the LLM's reply (text and/or tool calls)
    """
    client = _get_client()
    response = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=messages,
        tools=tools,
    )
    return response


# =============================================================================
# TOOL IMPLEMENTATIONS
# These are the actual functions that get executed when the LLM calls a tool.
# =============================================================================


def read(file_path: str) -> str:
    """Read and return the contents of a file."""
    with open(file_path, "r") as f:
        return f.read()


def write(file_path: str, content: str) -> str:
    """Write content to a file, creating directories if needed."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)
    return f"Successfully wrote to {file_path}"


def bash(command: str) -> str:
    """Execute a shell command and return stdout + stderr."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=BASH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {BASH_TIMEOUT}s: {command}"
    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    return output


# Map tool names to their implementations for easy lookup
TOOLS = {"read": read, "write": write, "bash": bash}

# ==========================================================================
# TOOL DEFINITIONS (JSON Schema)
# These tell the LLM what tools are available and how to call them.
# ==========================================================================
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write the contents to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


def _build_system_prompt():
    """Build a system prompt with working directory context."""
    return (
        "You are marco-code, a minimal coding assistant. "
        "You help users with software engineering tasks by reading files, "
        "writing files, and executing shell commands.\n\n"
        f"Working directory: {os.getcwd()}\n"
        f"Platform: {platform.system()} {platform.release()}\n"
    )


def main():
    p = argparse.ArgumentParser(
        prog="marco-code",
        description="A minimal LLM-powered coding assistant",
    )
    p.add_argument("-p", required=True, help="The prompt to send to the assistant")
    args = p.parse_args()

    # Initialize conversation with system context and user's prompt
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": args.p},
    ]

    # ==========================================================================
    # AGENT LOOP
    # This is the core pattern: repeatedly call the LLM until it stops.
    # Each iteration, the LLM either:
    #   1. Calls one or more tools (we execute them and continue)
    #   2. Responds with text and finish_reason="stop" (we exit the loop)
    # ==========================================================================
    for _iteration in range(MAX_ITERATIONS):
        chat = call_model(messages, TOOL_DEFINITIONS)

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("No choices in response")

        message = chat.choices[0].message
        messages.append(message)

        # If the LLM requested tool calls, execute them
        if message.tool_calls:
            for tool_call in message.tool_calls:
                fn = tool_call.function

                try:
                    fn_args = json.loads(fn.arguments)
                    print(f"  [tool] {fn.name}({fn_args})", file=sys.stderr)
                    result = TOOLS[fn.name](**fn_args)
                except (KeyError, json.JSONDecodeError, TypeError) as e:
                    result = f"Error: {type(e).__name__}: {e}"
                except subprocess.TimeoutExpired as e:
                    result = f"Error: command timed out after {BASH_TIMEOUT}s"
                except Exception as e:
                    result = f"Error executing {fn.name}: {e}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        # Exit loop when the LLM is done (no more tool calls)
        if chat.choices[0].finish_reason == "stop":
            break
    else:
        print(
            f"Warning: reached maximum iterations ({MAX_ITERATIONS})",
            file=sys.stderr,
        )

    # Print the final response
    final_content = chat.choices[0].message.content
    if final_content:
        print(final_content)
    else:
        print("Stopped: reached maximum iterations without a final response.")


if __name__ == "__main__":
    main()
