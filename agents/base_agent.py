"""
Base Agent - Abstract base class for all agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    WAITING = "waiting"


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    receiver: str
    content: Dict[str, Any]
    message_type: str = "request"  # request, response, error
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AgentResult:
    """Result returned by an agent"""
    agent_name: str
    status: AgentStatus
    data: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Each agent must implement:
    - process(): Main processing logic
    - validate_input(): Input validation
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.message_history: List[AgentMessage] = []
        self._initialized = False
        
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """Main processing method - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input data - must be implemented by subclasses"""
        pass
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the agent with necessary resources"""
        self._initialized = True
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent's task with timing and error handling.
        This is the main entry point called by the orchestrator.
        """
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        
        try:
            # Validate input
            is_valid, error_msg = self.validate_input(input_data)
            if not is_valid:
                self.status = AgentStatus.FAILED
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    data={},
                    errors=[f"Input validation failed: {error_msg}"],
                    execution_time=time.time() - start_time
                )
            
            # Process
            result = self.process(input_data)
            result.execution_time = time.time() - start_time
            
            self.status = result.status
            return result
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                data={},
                errors=[f"Execution error: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    def send_message(self, receiver: str, content: Dict[str, Any], 
                     message_type: str = "request") -> AgentMessage:
        """Create and log a message to another agent"""
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=message_type
        )
        self.message_history.append(message)
        return message
    
    def receive_message(self, message: AgentMessage):
        """Receive and log a message from another agent"""
        self.message_history.append(message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "name": self.name,
            "status": self.status.value,
            "initialized": self._initialized,
            "message_count": len(self.message_history)
        }
    
    def reset(self):
        """Reset agent state"""
        self.status = AgentStatus.IDLE
        self.message_history = []
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', status={self.status.value})"
