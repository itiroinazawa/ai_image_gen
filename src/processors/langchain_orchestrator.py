"""
Langchain orchestrator for AI Image & Video Generation Agent workflows.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, Tool
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent

from utils.config import Config

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Langchain-based workflow orchestrator for AI Image & Video Generation Agent.
    """

    def __init__(self, config: Config, llm: Optional[BaseLLM] = None):
        """
        Initialize the workflow orchestrator.

        Args:
            config: Configuration object
            llm: Language model for orchestration (optional)
        """
        self.config = config
        self.llm = llm
        self.memory = ConversationBufferMemory(return_messages=True)
        
    def create_image_generation_chain(
        self,
        image_generator_func: Callable,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> LLMChain:
        """
        Create a chain for image generation.

        Args:
            image_generator_func: Function to generate images
            default_params: Default parameters for image generation

        Returns:
            LLMChain for image generation
        """
        if default_params is None:
            default_params = {}
        
        # Create a prompt template for image generation
        prompt = PromptTemplate(
            input_variables=["prompt", "parameters"],
            template="Generate an image based on the following prompt: {prompt}\n\nParameters: {parameters}",
        )
        
        # Create a chain for image generation
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
        )
        
        return chain
    
    def create_image_processing_chain(
        self,
        image_processor_func: Callable,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> LLMChain:
        """
        Create a chain for image processing.

        Args:
            image_processor_func: Function to process images
            default_params: Default parameters for image processing

        Returns:
            LLMChain for image processing
        """
        if default_params is None:
            default_params = {}
        
        # Create a prompt template for image processing
        prompt = PromptTemplate(
            input_variables=["image_path", "prompt", "parameters"],
            template="Process the image at {image_path} based on the following prompt: {prompt}\n\nParameters: {parameters}",
        )
        
        # Create a chain for image processing
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
        )
        
        return chain
    
    def create_video_generation_chain(
        self,
        video_generator_func: Callable,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> LLMChain:
        """
        Create a chain for video generation.

        Args:
            video_generator_func: Function to generate videos
            default_params: Default parameters for video generation

        Returns:
            LLMChain for video generation
        """
        if default_params is None:
            default_params = {}
        
        # Create a prompt template for video generation
        prompt = PromptTemplate(
            input_variables=["prompt", "parameters"],
            template="Generate a video based on the following prompt: {prompt}\n\nParameters: {parameters}",
        )
        
        # Create a chain for video generation
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
        )
        
        return chain
    
    def create_agent(
        self,
        tools: List[Tool],
        agent_type: AgentType = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    ) -> AgentExecutor:
        """
        Create an agent for orchestrating workflows.

        Args:
            tools: List of tools for the agent to use
            agent_type: Type of agent to create

        Returns:
            Agent executor
        """
        # Initialize the agent
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=agent_type,
            memory=self.memory,
            verbose=True,
        )
        
        return agent
    
    def create_image_generation_tool(
        self,
        image_generator_func: Callable,
    ) -> Tool:
        """
        Create a tool for image generation.

        Args:
            image_generator_func: Function to generate images

        Returns:
            Tool for image generation
        """
        return Tool(
            name="ImageGenerator",
            func=image_generator_func,
            description="Generate an image from a text prompt",
        )
    
    def create_image_processing_tool(
        self,
        image_processor_func: Callable,
    ) -> Tool:
        """
        Create a tool for image processing.

        Args:
            image_processor_func: Function to process images

        Returns:
            Tool for image processing
        """
        return Tool(
            name="ImageProcessor",
            func=image_processor_func,
            description="Process an image using a text prompt",
        )
    
    def create_video_generation_tool(
        self,
        video_generator_func: Callable,
    ) -> Tool:
        """
        Create a tool for video generation.

        Args:
            video_generator_func: Function to generate videos

        Returns:
            Tool for video generation
        """
        return Tool(
            name="VideoGenerator",
            func=video_generator_func,
            description="Generate a video from a text prompt",
        )
    
    def create_workflow(
        self,
        image_generator_func: Optional[Callable] = None,
        image_processor_func: Optional[Callable] = None,
        video_generator_func: Optional[Callable] = None,
    ) -> AgentExecutor:
        """
        Create a workflow for image and video generation.

        Args:
            image_generator_func: Function to generate images
            image_processor_func: Function to process images
            video_generator_func: Function to generate videos

        Returns:
            Agent executor for the workflow
        """
        # Create tools for the agent
        tools = []
        
        if image_generator_func is not None:
            tools.append(self.create_image_generation_tool(image_generator_func))
        
        if image_processor_func is not None:
            tools.append(self.create_image_processing_tool(image_processor_func))
        
        if video_generator_func is not None:
            tools.append(self.create_video_generation_tool(video_generator_func))
        
        # Create the agent
        agent = self.create_agent(tools)
        
        return agent
