# LangChain

## What is LangChain?

LangChain is a framework for developing applications powered by language models. It provides:
- Abstraction layers for LLM interactions
- Tools for creating complex chains and flows
- Integration with various data sources and tools
- Memory management for conversations
- Built-in evaluation and testing tools

## Core Components of LangChain

1. **Models**: LLM and Chat Model interfaces
2. **Prompts**: Templates and management
3. **Chains**: Sequences of operations
4. **Agents**: Autonomous LLM-powered tools
5. **Memory**: State management for conversations
6. **Retrievers**: Data access interfaces
7. **Indexes**: Data structuring for LLM consumption

## How do you install LangChain?

```bash
# Basic installation
pip install langchain

# With specific extras
pip install langchain[all]  # All dependencies
pip install langchain[llms]  # Just LLM integrations
pip install langchain[embeddings]  # Just embedding integrations

# Common additional dependencies
pip install python-dotenv  # For environment variables
pip install openai  # For OpenAI integration
pip install chromadb  # For vector storage
```

## How do you use Language Models?

Basic LLM usage:
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Basic LLM
llm = OpenAI(temperature=0.7)
response = llm.predict("What is the capital of France?")

# Chat model
chat_model = ChatOpenAI(temperature=0.7)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]
response = chat_model.predict_messages(messages)
```

## How do you work with Prompts?

Prompt management examples:
```python
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

# Basic prompt template
prompt = PromptTemplate(
    input_variables=["product"],
    template="What are the best features of {product}?"
)

# Chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Tell me about {topic}"),
    ("assistant", "I'd be happy to help with that."),
    ("human", "Can you provide more details about {subtopic}?")
])

# Few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=[
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
    ],
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    ),
    prefix="Give the opposite of each input:",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)
```

## How do you create Chains?

Common chain examples:
```python
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

# Basic LLM Chain
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

# Sequential Chain
chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)
sequential_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

# Complex Sequential Chain
chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["initial_input"],
    output_variables=["final_output"],
    verbose=True
)

# Router Chain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain

prompt_infos = [
    {
        "name": "science",
        "description": "Good for answering science questions",
        "prompt_template": science_template
    },
    {
        "name": "history",
        "description": "Good for answering history questions",
        "prompt_template": history_template
    }
]

chain = MultiPromptChain(
    router_chain=LLMRouterChain.from_llm(llm),
    destination_chains={"science": science_chain, "history": history_chain},
    default_chain=default_chain,
    verbose=True
)
```

## How do you implement Memory?

Memory implementation examples:
```python
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory

# Basic Buffer Memory
memory = ConversationBufferMemory()
chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Window Memory (last N interactions)
memory = ConversationBufferWindowMemory(k=2)

# Summary Memory
memory = ConversationSummaryMemory(llm=llm)

# Vector Store Memory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

## How do you work with Agents?

Agent implementation examples:
```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# Load tools
tools = load_tools(
    ["python_repl", "requests_all", "terminal"],
    llm=llm
)

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Custom tool
from langchain.tools import Tool

def get_weather(location):
    """Get weather for location."""
    return f"Weather for {location}: Sunny"

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Useful for getting weather information"
)

# Agent with custom tools
agent = initialize_agent(
    [weather_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

## How do you handle Document Loading and Processing?

Document handling examples:
```python
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Load text document
loader = TextLoader("path/to/file.txt")
documents = loader.load()

# Load PDF
loader = PyPDFLoader("path/to/file.pdf")
pages = loader.load_and_split()

# Split text
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
splits = text_splitter.split_documents(documents)

# Process with embedding
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)
```

## How do you implement Retrievers?

Retriever implementation examples:
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Basic retriever
retriever = vectorstore.as_retriever()

# Contextual compression
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Self-query retriever
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_field_info = [
    DocumentMetadataFilter(
        name="source",
        description="The source of the document",
        type="string"
    ),
    DocumentMetadataFilter(
        name="date",
        description="The date of the document",
        type="date"
    ),
]

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_contents="Document contents here",
    metadata_field_info=metadata_field_info
)
```

## How do you handle Evaluation?

Evaluation examples:
```python
from langchain.evaluation import QAEvalChain
from langchain.evaluation import StringEvaluator

# Basic QA evaluation
examples = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris"
    }
]
eval_chain = QAEvalChain.from_llm(llm)
results = eval_chain.evaluate(
    examples,
    predictions=predictions
)

# String evaluation
evaluator = StringEvaluator()
score = evaluator.evaluate_strings(
    prediction="The capital is Paris",
    reference="Paris is the capital"
)

# Custom evaluation
from langchain.evaluation import load_evaluator

evaluator = load_evaluator(
    "criteria",
    criteria={
        "accuracy": "Is the response accurate?",
        "clarity": "Is the response clear?"
    }
)
eval_result = evaluator.evaluate_strings(
    prediction="Response here",
    input="Question here"
)
```

## How do you implement Caching?

Caching implementation:
```python
from langchain.cache import InMemoryCache
from langchain.cache import SQLiteCache
import langchain

# In-memory cache
langchain.cache = InMemoryCache()

# SQLite cache
langchain.cache = SQLiteCache(database_path=".langchain.db")

# Redis cache
from langchain.cache import RedisCache
import redis

redis_client = redis.Redis.from_url("redis://localhost:6379")
langchain.cache = RedisCache(redis_client)
```

## Best Practices and Tips

1. **Environment Management**:
```python
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

2. **Error Handling**:
```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = chain.run(input)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost}")
```

3. **Debugging**:
```python
# Enable verbose mode
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

# Custom callbacks
from langchain.callbacks import StdOutCallbackHandler

handler = StdOutCallbackHandler()
chain.run(input, callbacks=[handler])
```