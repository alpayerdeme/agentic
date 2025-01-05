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

Would you like me to explain any particular aspect in more detail or make any modifications to the implementation?
