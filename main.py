#simple ReAct agent with tools (Ollama)

from __future__ import annotations
import ast
import operator as op
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor


# -------- Tools --------
# tools are small functions the agent can use.
#made using @tools decorator
#https://python.langchain.com/docs/how_to/custom_tools/?utm_source=chatgpt.com

@tool("calculator", return_direct=True)
def calculator(expr: str) -> str:
    """Do some safe arithmetic"""

    allowed_ops = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.USub: op.neg
    }
    allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Pow)

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported syntax.")

    tree = ast.parse(expr, mode="eval")
    for n in ast.walk(tree):
        if not isinstance(n, allowed_nodes) and type(n) not in allowed_ops:
            raise ValueError(f"Unsupported: {type(n).__name__}")
    return str(_eval(tree.body))


@tool("melbourne_time", return_direct=False)
def melbourne_time(_: str = "") -> str:
    """get current date and time in melbourne."""
    tz = ZoneInfo("Australia/Melbourne")
    now = datetime.now(tz)
    return now.strftime("%A, %d %B %Y %H:%M:%S %Z")


# -------- LLM + Prompt --------
# this is were LLM, in my case llama3.2 comes in 

llm = ChatOllama(model="llama3.2:latest", temperature=0.2)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You can use these tools:\n{tools}\n\n"
         "Tools: {tool_names}\n\n"
         "Follow this format:\n"
         "Thought: <reasoning>\n"
         "Action: <tool>\n"
         "Action Input: <input>\n"
         "Observation: <result>\n"
         "Final Answer: <answer>"),
        ("human", "What is 2+3?"),
        ("ai", "Thought: I should use calculator.\nAction: calculator\nAction Input: 2+3"),
        ("ai", "Thought: I now know.\nFinal Answer: 5"),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ]
)


# -------- Build the agent --------
tools = [calculator, melbourne_time]
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4,
    early_stopping_method="generate",
)


# test questions 

if __name__ == "__main__":
    print(agent_executor.invoke({"input": "What time is it in Melbourne? Use the right tool."})["output"])
    print(agent_executor.invoke({"input": "Please compute (8+3)*4 - 5**2 / 5."})["output"])
