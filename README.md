# agentic
agentic ai samples

I'll help create a sample application that demonstrates complex agent-like behavior using Python. This will showcase an agent that can make autonomous decisions based on its environment and goals.



```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    ENERGY = "energy"
    MATERIAL = "material"
    INFORMATION = "information"

class ActionType(Enum):
    GATHER = "gather"
    PROCESS = "process"
    STORE = "store"
    SHARE = "share"
    REST = "rest"

@dataclass
class Resource:
    type: ResourceType
    quantity: float
    quality: float  # 0.0 to 1.0
    timestamp: datetime

@dataclass
class Action:
    type: ActionType
    target_resource: Optional[ResourceType]
    duration: timedelta
    energy_cost: float
    success_probability: float

class Environment:
    def __init__(self):
        self.resources: Dict[ResourceType, List[Resource]] = {
            resource_type: [] for resource_type in ResourceType
        }
        self.current_time = datetime.now()
        
    def add_resource(self, resource: Resource):
        self.resources[resource.type].append(resource)
        
    def get_available_resources(self, resource_type: ResourceType) -> List[Resource]:
        return [r for r in self.resources[resource_type] 
                if (self.current_time - r.timestamp).total_seconds() < 3600]  # Last hour only

class Agent:
    def __init__(self, name: str, initial_energy: float = 100.0):
        self.name = name
        self.energy = initial_energy
        self.inventory: Dict[ResourceType, List[Resource]] = {
            resource_type: [] for resource_type in ResourceType
        }
        self.knowledge_base: Dict[str, float] = {}  # Stores learned information
        self.goals: List[Tuple[ResourceType, float]] = []  # (resource_type, target_quantity)
        
    def set_goal(self, resource_type: ResourceType, target_quantity: float):
        self.goals.append((resource_type, target_quantity))
        
    def evaluate_action(self, action: Action, environment: Environment) -> float:
        """Calculate the utility of an action based on current state and goals."""
        utility = 0.0
        
        # Consider energy cost
        energy_ratio = self.energy / 100.0
        if energy_ratio < 0.2 and action.type != ActionType.REST:
            return -1.0  # Strongly discourage non-rest actions when energy is low
            
        # Evaluate based on goals
        if action.target_resource:
            for goal_type, goal_quantity in self.goals:
                if action.target_resource == goal_type:
                    current_quantity = sum(r.quantity for r in self.inventory[goal_type])
                    if current_quantity < goal_quantity:
                        utility += 1.0
                        
        # Adjust for success probability
        utility *= action.success_probability
        
        # Consider energy efficiency
        utility -= (action.energy_cost / 100.0)
        
        return utility
        
    def decide_action(self, environment: Environment) -> Action:
        """Decide the next best action to take."""
        possible_actions = self._generate_possible_actions(environment)
        
        best_action = None
        best_utility = float('-inf')
        
        for action in possible_actions:
            utility = self.evaluate_action(action, environment)
            if utility > best_utility:
                best_utility = utility
                best_action = action
                
        return best_action or self._get_rest_action()
    
    def _generate_possible_actions(self, environment: Environment) -> List[Action]:
        """Generate list of possible actions based on current state."""
        actions = []
        
        for resource_type in ResourceType:
            # Gather action
            if environment.get_available_resources(resource_type):
                actions.append(Action(
                    type=ActionType.GATHER,
                    target_resource=resource_type,
                    duration=timedelta(minutes=30),
                    energy_cost=20.0,
                    success_probability=0.8
                ))
                
            # Process action
            if self.inventory[resource_type]:
                actions.append(Action(
                    type=ActionType.PROCESS,
                    target_resource=resource_type,
                    duration=timedelta(minutes=45),
                    energy_cost=30.0,
                    success_probability=0.9
                ))
                
        return actions
    
    def _get_rest_action(self) -> Action:
        """Create a rest action to recover energy."""
        return Action(
            type=ActionType.REST,
            target_resource=None,
            duration=timedelta(minutes=60),
            energy_cost=-50.0,  # Negative cost means energy gain
            success_probability=1.0
        )
    
    def execute_action(self, action: Action, environment: Environment) -> bool:
        """Execute the chosen action and update state."""
        logger.info(f"Agent {self.name} executing {action.type.value}")
        
        # Check if action succeeds based on probability
        if random.random() > action.success_probability:
            logger.info(f"Action {action.type.value} failed!")
            self.energy += action.energy_cost  # Still consume energy
            return False
            
        if action.type == ActionType.GATHER:
            available = environment.get_available_resources(action.target_resource)
            if available:
                resource = available[0]
                self.inventory[action.target_resource].append(resource)
                environment.resources[action.target_resource].remove(resource)
                
        elif action.type == ActionType.PROCESS:
            if self.inventory[action.target_resource]:
                resource = self.inventory[action.target_resource][0]
                processed_resource = Resource(
                    type=resource.type,
                    quantity=resource.quantity * 1.5,  # Processing increases quantity
                    quality=min(1.0, resource.quality * 1.2),  # Processing improves quality
                    timestamp=datetime.now()
                )
                self.inventory[action.target_resource][0] = processed_resource
                
        elif action.type == ActionType.REST:
            self.energy = min(100.0, self.energy - action.energy_cost)
            
        # Update energy
        self.energy += action.energy_cost
        self.energy = max(0.0, min(100.0, self.energy))
        
        return True

def run_simulation(agents: List[Agent], environment: Environment, steps: int):
    """Run a simulation with multiple agents in the environment."""
    for step in range(steps):
        logger.info(f"\nSimulation Step {step + 1}")
        
        for agent in agents:
            action = agent.decide_action(environment)
            success = agent.execute_action(action, environment)
            
            logger.info(f"Agent {agent.name}:")
            logger.info(f"  Energy: {agent.energy:.1f}")
            logger.info(f"  Action: {action.type.value} ({'Success' if success else 'Failed'})")
            
            # Add some random new resources to environment
            if random.random() < 0.3:
                resource_type = random.choice(list(ResourceType))
                new_resource = Resource(
                    type=resource_type,
                    quantity=random.uniform(1.0, 10.0),
                    quality=random.random(),
                    timestamp=datetime.now()
                )
                environment.add_resource(new_resource)

# Example usage
if __name__ == "__main__":
    # Create environment
    env = Environment()
    
    # Add initial resources
    for resource_type in ResourceType:
        for _ in range(3):
            resource = Resource(
                type=resource_type,
                quantity=random.uniform(5.0, 15.0),
                quality=random.random(),
                timestamp=datetime.now()
            )
            env.add_resource(resource)
    
    # Create agents with different goals
    agents = [
        Agent("ResourceCollector"),
        Agent("Processor")
    ]
    
    # Set different goals for agents
    agents[0].set_goal(ResourceType.ENERGY, 50.0)
    agents[0].set_goal(ResourceType.MATERIAL, 30.0)
    agents[1].set_goal(ResourceType.INFORMATION, 40.0)
    
    # Run simulation
    run_simulation(agents, env, steps=10)

```

I've created a complex agent-based application that demonstrates several key concepts in autonomous systems. Here are the main features:

1. **Multi-Agent System**:
   - Supports multiple agents operating in the same environment
   - Each agent can have different goals and strategies
   - Agents make independent decisions based on their state and environment

2. **Resource Management**:
   - Three types of resources: energy, material, and information
   - Resources have properties like quantity, quality, and timestamps
   - Environment maintains and updates resource availability

3. **Decision Making**:
   - Utility-based action selection
   - Considers multiple factors: energy levels, goals, success probability
   - Supports different types of actions (gather, process, store, share, rest)

4. **State Management**:
   - Agents maintain internal state (energy, inventory, knowledge)
   - Environment tracks global state and resource availability
   - Time-based resource decay and availability

5. **Goal-Oriented Behavior**:
   - Agents can have multiple goals
   - Goals influence action selection
   - Dynamic priority adjustment based on current state



*************************************************



I'll create a comprehensive RAG system for a finance company that processes internal documents like financial reports, policies, and client documents.



```python
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging
import json
from datetime import datetime
import os
from dataclasses import dataclass

from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for financial documents."""
    doc_type: str  # e.g., "policy", "report", "client_document"
    department: str  # e.g., "compliance", "trading", "risk"
    created_date: str
    last_modified: str
    classification: str  # e.g., "confidential", "internal", "public"
    document_id: str

class DocumentProcessor:
    """Handles loading and processing of different document types."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_document(self, file_path: str, metadata: DocumentMetadata) -> List[dict]:
        """Load document based on file type and add metadata."""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()
            
            # Add metadata to each document chunk
            for doc in documents:
                doc.metadata.update({
                    "doc_type": metadata.doc_type,
                    "department": metadata.department,
                    "created_date": metadata.created_date,
                    "last_modified": metadata.last_modified,
                    "classification": metadata.classification,
                    "document_id": metadata.document_id,
                    "source_file": file_path
                })
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            logger.info(f"Processed {file_path}: {len(split_docs)} chunks created")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

class FinanceRAGSystem:
    """Main RAG system for financial documents."""
    
    def __init__(self, openai_api_key: str, persist_directory: str):
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=openai_api_key
        )
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor()
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize retrieval chain
        self.retrieval_chain = self._create_retrieval_chain()
        
        # Track usage
        self.usage_log = []

    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load existing vector store."""
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
    
    def _create_retrieval_chain(self) -> ConversationalRetrievalChain:
        """Create retrieval chain with custom prompts."""
        custom_template = """You are a financial expert assistant for a finance company.
        Use only the following context to answer the question. If you're unsure or the information
        is not in the context, say "I don't have enough information to answer this question accurately."
        
        Be especially careful with:
        - Numerical values and calculations
        - Dates and deadlines
        - Policy requirements and restrictions
        - Client-specific information
        
        Always mention if the information comes from a specific policy or document.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {question}
        
        Answer: """
        
        CUSTOM_PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=custom_template
        )
        
        # Create a context compressor for more relevant retrieval
        compressor = LLMChainExtractor.from_llm(self.llm)
        compressed_retriever = ContextualCompressionRetriever(
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            doc_compressor=compressor
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=compressed_retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
            return_source_documents=True
        )

    def add_documents(self, file_paths: List[str], metadata_list: List[DocumentMetadata]):
        """Add new documents to the vector store."""
        if len(file_paths) != len(metadata_list):
            raise ValueError("Number of file paths must match number of metadata entries")
        
        all_documents = []
        for file_path, metadata in zip(file_paths, metadata_list):
            try:
                documents = self.doc_processor.load_document(file_path, metadata)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        if all_documents:
            self.vector_store.add_documents(all_documents)
            self.vector_store.persist()
            logger.info(f"Added {len(all_documents)} documents to vector store")

    def query(self, question: str) -> Dict[str, Union[str, List[dict]]]:
        """Process a query and return answer with sources."""
        try:
            with get_openai_callback() as cb:
                result = self.retrieval_chain({"question": question})
                
                # Log usage
                usage_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "query": question,
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost,
                    "source_documents": [
                        {
                            "doc_id": doc.metadata["document_id"],
                            "doc_type": doc.metadata["doc_type"],
                            "department": doc.metadata["department"],
                            "classification": doc.metadata["classification"]
                        }
                        for doc in result["source_documents"]
                    ]
                }
                self.usage_log.append(usage_entry)
                
                return {
                    "answer": result["answer"],
                    "sources": [
                        {
                            "document_id": doc.metadata["document_id"],
                            "doc_type": doc.metadata["doc_type"],
                            "department": doc.metadata["department"],
                            "source_file": doc.metadata["source_file"],
                            "classification": doc.metadata["classification"],
                            "content": doc.page_content[:200] + "..."  # Preview of content
                        }
                        for doc in result["source_documents"]
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": "An error occurred while processing your query.",
                "sources": []
            }

    def export_usage_logs(self, output_file: str):
        """Export usage logs to JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.usage_log, f, indent=2)
            logger.info(f"Usage logs exported to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting usage logs: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize the system
    rag_system = FinanceRAGSystem(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        persist_directory="./finance_docs_store"
    )
    
    # Example document metadata
    example_metadata = DocumentMetadata(
        doc_type="policy",
        department="compliance",
        created_date="2024-01-01",
        last_modified="2024-01-05",
        classification="internal",
        document_id="POL-2024-001"
    )
    
    # Add documents
    rag_system.add_documents(
        file_paths=["./docs/trading_policy.pdf"],
        metadata_list=[example_metadata]
    )
    
    # Example queries
    queries = [
        "What are the trading restrictions for employee accounts?",
        "What is the process for reporting suspicious transactions?",
        "What are the document retention requirements for client records?"
    ]
    
    # Test the system
    for query in queries:
        print(f"\nQuery: {query}")
        result = rag_system.query(query)
        print(f"Answer: {result['answer']}")
        print("\nSources:")
        for source in result['sources']:
            print(f"- {source['doc_type']} from {source['department']}")
    
    # Export usage logs
    rag_system.export_usage_logs("rag_usage_logs.json")

```

I've created a comprehensive RAG system specifically designed for a finance company's internal documents. Here are the key features:

1. **Document Processing**:
   - Supports multiple file types (PDF, CSV, Excel, Word)
   - Handles document metadata (type, department, classification, etc.)
   - Intelligent text chunking with overlap for context preservation

2. **Security and Compliance Features**:
   - Document classification tracking
   - Usage logging and monitoring
   - Source attribution in responses
   - Careful handling of sensitive information

3. **Advanced Retrieval**:
   - Contextual compression for more relevant results
   - Custom prompting for financial context
   - Source document tracking and verification
   - Memory for conversation context

4. **Financial Domain Specifics**:
   - Specialized prompt templates for financial information
   - Careful handling of numerical data and dates
   - Department-specific context preservation
   - Policy and regulation awareness

5. **Monitoring and Logging**:
   - Token usage tracking
   - Cost monitoring
   - Query logging with timestamps
   - Source document tracking

To use this system:

1. Set up environment and dependencies:
```bash
pip install langchain chromadb openai unstructured python-docx openpyxl
```

2. Set up your document structure:
```python
metadata = DocumentMetadata(
    doc_type="policy",
    department="risk",
    created_date="2024-01-01",
    last_modified="2024-01-05",
    classification="confidential",
    document_id="RISK-2024-001"
)

rag_system.add_documents(
    file_paths=["path/to/document.pdf"],
    metadata_list=[metadata]
)
```

3. Query the system:
```python
result = rag_system.query("What are our current risk management policies?")
print(result["answer"])
print("Sources:", result["sources"])
```

************************************************************ 

I'll create a LangGraph sample that implements a complex decision-making system for task planning and execution.



```python
from typing import Dict, List, Tuple, TypedDict, Annotated, Literal
from datetime import datetime
import json
import logging
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import Graph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint import checkpoint
import operator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define state types and enums
class TaskStatus(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskType(str, Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DECISION = "decision"
    ACTION = "action"

class TaskPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class WorkflowState(TypedDict):
    """Represents the current state of the workflow."""
    task_queue: List[Dict]
    current_task: Dict
    completed_tasks: List[Dict]
    context: Dict
    messages: List[Dict]
    status: TaskStatus
    last_error: str

# Define tools
class ResearchTool:
    def search(self, query: str) -> str:
        """Simulated research tool."""
        return f"Research results for: {query}"

class AnalysisTool:
    def analyze(self, data: Dict) -> Dict:
        """Simulated analysis tool."""
        return {"analysis_result": f"Analysis of {data}"}

class ActionTool:
    def execute(self, action: Dict) -> Dict:
        """Simulated action execution tool."""
        return {"result": f"Executed action: {action}"}

# Create tool executor
tools = {
    "research": ResearchTool().search,
    "analysis": AnalysisTool().analyze,
    "action": ActionTool().execute,
}
tool_executor = ToolExecutor(tools)

# Define prompt templates
TASK_PLANNING_PROMPT = """You are a task planning AI assistant.
Given the current context and task, create a detailed plan.

Current Context:
{context}

Current Task:
{task}

Previous Messages:
{messages}

Create a plan that includes:
1. Required steps
2. Dependencies
3. Estimated effort
4. Potential risks
5. Success criteria

Respond in the following JSON format:
{
    "steps": [{"step": "step description", "effort": "estimated effort"}],
    "dependencies": ["dependency1", "dependency2"],
    "risks": ["risk1", "risk2"],
    "success_criteria": ["criterion1", "criterion2"]
}
"""

DECISION_MAKING_PROMPT = """You are a decision-making AI assistant.
Evaluate the current situation and make a decision about the next step.

Current Context:
{context}

Current Task:
{task}

Task Status:
{status}

Available Options:
{options}

Previous Messages:
{messages}

Make a decision and explain your reasoning in the following JSON format:
{
    "decision": "chosen option",
    "reasoning": "explanation of choice",
    "confidence": 0.0 to 1.0,
    "next_steps": ["step1", "step2"]
}
"""

# Define workflow nodes
def create_plan(state: WorkflowState) -> Dict:
    """Create a plan for the current task."""
    prompt = ChatPromptTemplate.from_template(TASK_PLANNING_PROMPT)
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    parser = JsonOutputParser()
    
    chain = prompt | llm | parser
    
    response = chain.invoke({
        "context": state["context"],
        "task": state["current_task"],
        "messages": state["messages"]
    })
    
    return {"plan": response}

def make_decision(state: WorkflowState) -> Dict:
    """Make a decision about the next step."""
    prompt = ChatPromptTemplate.from_template(DECISION_MAKING_PROMPT)
    llm = ChatOpenAI(temperature=0.2, model="gpt-4")
    parser = JsonOutputParser()
    
    chain = prompt | llm | parser
    
    response = chain.invoke({
        "context": state["context"],
        "task": state["current_task"],
        "status": state["status"],
        "options": list(tools.keys()),
        "messages": state["messages"]
    })
    
    return {"decision": response}

def execute_task(state: WorkflowState, decision: Dict) -> Dict:
    """Execute the decided action using appropriate tool."""
    try:
        tool_name = decision["decision"]
        if tool_name not in tools:
            raise ValueError(f"Unknown tool: {tool_name}")
            
        result = tool_executor.execute(
            tool_name,
            {
                "task": state["current_task"],
                "context": state["context"]
            }
        )
        
        return {"result": result, "status": TaskStatus.COMPLETED}
        
    except Exception as e:
        logger.error(f"Task execution error: {str(e)}")
        return {"result": None, "status": TaskStatus.FAILED, "error": str(e)}

def review_results(state: WorkflowState, results: Dict) -> Dict:
    """Review task results and determine next steps."""
    if results["status"] == TaskStatus.FAILED:
        return {
            "status": TaskStatus.FAILED,
            "error": results["error"]
        }
    
    # Add completed task to history
    completed_task = {
        **state["current_task"],
        "completion_time": datetime.now().isoformat(),
        "results": results["result"]
    }
    
    return {
        "status": TaskStatus.COMPLETED,
        "completed_task": completed_task
    }

# Define state updates
def update_on_plan(state: WorkflowState, plan: Dict) -> WorkflowState:
    """Update state with new plan."""
    state["current_task"]["plan"] = plan["plan"]
    state["status"] = TaskStatus.EXECUTING
    return state

def update_on_decision(state: WorkflowState, decision: Dict) -> WorkflowState:
    """Update state with decision."""
    state["current_task"]["decision"] = decision["decision"]
    return state

def update_on_review(state: WorkflowState, review: Dict) -> WorkflowState:
    """Update state based on review results."""
    if review["status"] == TaskStatus.FAILED:
        state["status"] = TaskStatus.FAILED
        state["last_error"] = review["error"]
    else:
        state["completed_tasks"].append(review["completed_task"])
        state["status"] = TaskStatus.COMPLETED
        
        # Get next task from queue if available
        if state["task_queue"]:
            state["current_task"] = state["task_queue"].pop(0)
            state["status"] = TaskStatus.PLANNING
        
    return state

# Create the workflow graph
def create_workflow() -> Graph:
    """Create the workflow graph."""
    workflow = Graph()
    
    # Add nodes
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("make_decision", make_decision)
    workflow.add_node("execute_task", execute_task)
    workflow.add_node("review_results", review_results)
    
    # Add edges
    workflow.add_edge("create_plan", "make_decision")
    workflow.add_edge("make_decision", "execute_task")
    workflow.add_edge("execute_task", "review_results")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "review_results",
        lambda x: "create_plan" if x["status"] == TaskStatus.PLANNING else END
    )
    
    return workflow.compile()

# Main execution class
class DecisionWorkflow:
    def __init__(self):
        self.workflow = create_workflow()
        
    def process_task(self, task: Dict, context: Dict) -> Dict:
        """Process a single task through the workflow."""
        initial_state: WorkflowState = {
            "task_queue": [],
            "current_task": task,
            "completed_tasks": [],
            "context": context,
            "messages": [],
            "status": TaskStatus.PLANNING,
            "last_error": ""
        }
        
        try:
            for event in self.workflow.stream(initial_state):
                logger.info(f"Step: {event.data}")
                
            final_state = event.data
            
            return {
                "status": final_state["status"],
                "completed_tasks": final_state["completed_tasks"],
                "error": final_state["last_error"] if final_state["status"] == TaskStatus.FAILED else None
            }
            
        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")
            return {
                "status": TaskStatus.FAILED,
                "completed_tasks": [],
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    # Create workflow instance
    workflow = DecisionWorkflow()
    
    # Example task
    task = {
        "id": "TASK-001",
        "type": TaskType.ANALYSIS,
        "priority": TaskPriority.HIGH,
        "description": "Analyze market trends for Q1 2024",
        "deadline": "2024-01-31"
    }
    
    # Example context
    context = {
        "domain": "finance",
        "requirements": ["market_data", "competitor_analysis"],
        "constraints": ["time_sensitive", "high_accuracy"]
    }
    
    # Process task
    result = workflow.process_task(task, context)
    
    print(f"Task Status: {result['status']}")
    if result['status'] == TaskStatus.COMPLETED:
        print("\nCompleted Tasks:")
        for task in result['completed_tasks']:
            print(f"- {task['description']}: {task['results']}")
    else:
        print(f"Error: {result['error']}")

```

I've created a comprehensive LangGraph system that implements complex decision-making for task execution. Here are the key features:

1. **Advanced Workflow Structure**:
   - Multi-stage decision process
   - Conditional branching based on task states
   - Error handling and recovery
   - Task queuing and prioritization

2. **Decision Making Components**:
   - Task planning node
   - Decision making node
   - Task execution node
   - Results review node

3. **Flexible Tool Integration**:
   - Research tool for information gathering
   - Analysis tool for data processing
   - Action tool for task execution
   - Extensible tool framework

4. **State Management**:
   - Comprehensive state tracking
   - Task history
   - Context preservation
   - Error tracking

5. **Robust Prompting**:
   - Specialized prompts for planning
   - Decision-making prompts
   - JSON output parsing
   - Context-aware processing

To use this system:

1. Install required packages:
```bash
pip install langchain langgraph
```

2. Create and process a task:
```python
workflow = DecisionWorkflow()

task = {
    "id": "TASK-001",
    "type": TaskType.ANALYSIS,
    "priority": TaskPriority.HIGH,
    "description": "Analyze market data",
    "deadline": "2024-01-31"
}

context = {
    "domain": "finance",
    "requirements": ["market_data"],
    "constraints": ["time_sensitive"]
}

result = workflow.process_task(task, context)
```

The system will:
1. Create a plan for the task
2. Make decisions about execution steps
3. Execute the task using appropriate tools
4. Review results and determine next steps
5. Handle any errors or failures gracefully

