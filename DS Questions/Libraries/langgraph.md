# LangGraph

## What is LangGraph?

LangGraph is a library for building stateful, multi-agent workflows using LangChain. It provides:
- Tools for creating complex agent interactions
- State management capabilities
- Graph-based workflow definitions
- Integration with LangChain's components
- Support for cyclic workflows
- Concurrent agent execution

## Core Concepts in LangGraph

1. **Graph**: The workflow definition
2. **Node**: Individual steps in the workflow
3. **Edge**: Connections between nodes
4. **State**: Workflow context and data
5. **Channel**: Communication pathway between agents
6. **Agent**: LLM-powered worker

## How do you install LangGraph?

```bash
# Basic installation
pip install langgraph

# With LangChain
pip install langgraph langchain

# Additional dependencies
pip install openai  # For OpenAI integration
pip install networkx  # For graph visualization
```

## How do you create a Basic Graph?

Basic graph creation:
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from typing import Sequence

# Define state type
class GraphState(TypedDict):
    messages: Sequence[str]
    current_step: str

# Create graph
workflow = StateGraph(GraphState)

# Add node
def process_step(state):
    # Process state
    return {"messages": state["messages"] + ["Processed"]}

workflow.add_node("process", process_step)

# Add edge
workflow.add_edge("process", END)

# Compile graph
chain = workflow.compile()
```

## How do you implement Multi-Agent Workflows?

Multi-agent implementation:
```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

# Create agents
researcher = create_react_agent(
    llm=ChatOpenAI(temperature=0),
    tools=[
        Tool(
            name="search",
            func=lambda x: "search results",
            description="Search for information"
        )
    ],
    prompt_template="You are a researcher. {input}"
)

writer = create_react_agent(
    llm=ChatOpenAI(temperature=0.7),
    tools=[],
    prompt_template="You are a writer. {input}"
)

# Create graph
class AgentState(TypedDict):
    messages: list
    next_agent: str

workflow = StateGraph(AgentState)

# Add nodes
def researcher_node(state):
    response = researcher.invoke(state["messages"][-1])
    return {"messages": state["messages"] + [response], "next_agent": "writer"}

def writer_node(state):
    response = writer.invoke(state["messages"][-1])
    return {"messages": state["messages"] + [response], "next_agent": END}

workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

# Add edges
workflow.add_conditional_edges(
    "researcher",
    lambda x: x["next_agent"],
    {
        "writer": "writer",
        END: END
    }
)

workflow.add_conditional_edges(
    "writer",
    lambda x: x["next_agent"],
    {
        "researcher": "researcher",
        END: END
    }
)
```

## How do you handle State Management?

State management examples:
```python
from typing import Dict, List

# Define state types
class ConversationState(TypedDict):
    messages: List[str]
    agent_states: Dict[str, dict]
    memory: Dict[str, list]

# Initialize state
initial_state = ConversationState(
    messages=[],
    agent_states={},
    memory={}
)

# State update function
def update_state(state: ConversationState, update: dict) -> ConversationState:
    return {
        **state,
        **update
    }

# State validation
def validate_state(state: ConversationState) -> bool:
    required_keys = {"messages", "agent_states", "memory"}
    return all(key in state for key in required_keys)

# State persistence
def save_state(state: ConversationState, filepath: str):
    import json
    with open(filepath, "w") as f:
        json.dump(state, f)

def load_state(filepath: str) -> ConversationState:
    import json
    with open(filepath, "r") as f:
        return json.load(f)
```

## How do you implement Channels?

Channel implementation:
```python
from langgraph.channels import Channel
from typing import AsyncIterator

# Create channel
channel = Channel()

# Producer function
async def produce_messages() -> AsyncIterator[str]:
    messages = ["Message 1", "Message 2", "Message 3"]
    for message in messages:
        await channel.put(message)
        yield message

# Consumer function
async def consume_messages():
    async for message in channel:
        print(f"Received: {message}")

# Broadcast channel
broadcast_channel = Channel(broadcast=True)

# Channel with filtering
filtered_channel = Channel(
    filter_func=lambda message: message.startswith("important:")
)
```

## How do you handle Concurrent Execution?

Concurrent execution examples:
```python
from langgraph.concurrent import TaskGroup

# Parallel agent execution
async def run_agents_parallel(agents, input_data):
    async with TaskGroup() as group:
        tasks = []
        for agent in agents:
            task = group.create_task(agent.arun(input_data))
            tasks.append(task)
    
    results = [task.result() for task in tasks]
    return results

# Rate limiting
from langgraph.concurrent import RateLimiter

rate_limiter = RateLimiter(
    max_calls=60,
    time_window=60  # 60 calls per minute
)

async def rate_limited_execution():
    async with rate_limiter:
        result = await agent.arun(input_data)
    return result
```

## How do you implement Error Handling?

Error handling examples:
```python
from langgraph.errors import NodeError, GraphError

# Node error handling
def safe_node_execution(state):
    try:
        result = process_node(state)
        return result
    except Exception as e:
        raise NodeError(f"Error in node: {str(e)}")

# Graph error handling
try:
    result = workflow.run(initial_state)
except GraphError as e:
    print(f"Graph execution failed: {str(e)}")
    # Implement recovery logic

# Retry mechanism
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def retry_node(state):
    return process_node(state)
```

## How do you implement Monitoring and Logging?

Monitoring implementation:
```python
from langgraph.callbacks import BaseCallbackHandler
import logging

# Custom callback handler
class MonitoringCallback(BaseCallbackHandler):
    def on_node_start(self, node: str, inputs: dict):
        logging.info(f"Node {node} started with inputs: {inputs}")
    
    def on_node_end(self, node: str, outputs: dict):
        logging.info(f"Node {node} completed with outputs: {outputs}")
    
    def on_error(self, error: Exception):
        logging.error(f"Error occurred: {str(error)}")

# Add monitoring to graph
workflow.add_callback(MonitoringCallback())

# Metrics collection
from prometheus_client import Counter, Histogram

node_duration = Histogram(
    'node_duration_seconds',
    'Time spent in node execution',
    ['node_name']
)

node_errors = Counter(
    'node_errors_total',
    'Total number of node errors',
    ['node_name']
)
```

## How do you implement Testing?

Testing examples:
```python
import pytest
from unittest.mock import Mock

# Test node function
def test_node_execution():
    initial_state = {"messages": []}
    result = process_node(initial_state)
    assert "messages" in result
    assert len(result["messages"]) > 0

# Test graph execution
def test_workflow():
    mock_agent = Mock()
    mock_agent.invoke.return_value = "Response"
    
    workflow = create_workflow(mock_agent)
    result = workflow.run({"messages": ["Input"]})
    
    assert mock_agent.invoke.called
    assert "Response" in result["messages"]

# Test state transitions
def test_state_transitions():
    workflow = create_workflow()
    initial_state = {"messages": [], "current_step": "start"}
    
    result = workflow.run(initial_state)
    assert result["current_step"] == "end"
```

## Best Practices and Tips

1. **Graph Design**:
```python
# Break down complex workflows
def create_subgraph():
    subgraph = StateGraph(SubGraphState)
    # Add nodes and edges
    return subgraph.compile()

main_workflow.add_node("subprocess", create_subgraph())

# Document graph structure
workflow.add_node("step1", step1_function, description="Processes initial input")
workflow.add_node("step2", step2_function, description="Transforms data")
```

2. **Performance Optimization**:
```python
# Caching
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_node_operation(input_data: str) -> str:
    # Expensive processing
    return result

# Batch processing
def batch_process_node(states: List[dict]) -> List[dict]:
    # Process multiple states at once
    return processed_states
```

3. **Debugging**:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug nodes
def debug_node(state):
    print(f"Current state: {state}")
    return state

workflow.add_node("debug", debug_node)
```