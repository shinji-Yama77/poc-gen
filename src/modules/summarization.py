"""
Summarization Module.
Handles summarizing meeting transcripts into potential ideas.
"""
import logging
import uuid
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from src.core.llm_factory import LLMFactory, LLMConfig

logger = logging.getLogger("summarization")

class SummarizationModule:
    """
    Module for summarizing meeting transcripts into potential ideas.
    Uses LLMs to analyze the transcript and extract key ideas.
    """
    
    def __init__(self, config: Optional[dict] = None, llm: Optional[BaseChatModel] = None):
        """
        Initialize the summarization module with configuration.
        
        Args:
            config: Optional configuration for summarization
            llm: Optional pre-configured LLM instance (provided by orchestrator)
        """
        self.config = config or {}
        
        # Use provided LLM if available
        if llm:
            self.llm = llm
            logger.info("Using externally provided LLM instance for summarization")
            return
        
        # Otherwise, create own LLM instance using factory (for backward compatibility)
        logger.info("Creating new LLM instance for summarization")
        
        # Convert config to LLMConfig if needed
        if isinstance(config, dict):
            llm_config = LLMConfig(
                ai_provider=config.get("ai_provider", "openai"),
                temperature=config.get("temperature", 0.3),
                api_key=config.get("api_key"),
                model=config.get("model", "gpt-4"),
                azure_api_key=config.get("azure_api_key"),
                azure_endpoint=config.get("azure_endpoint"),
                azure_deployment=config.get("azure_deployment"),
                azure_api_version=config.get("azure_api_version"),
            )
        else:
            llm_config = LLMConfig.from_env()
            # Use gpt-4 as default model for summarization
            if llm_config.model == "gpt-4o":
                llm_config.model = "gpt-4"
        
        # Create LLM using factory
        try:
            self.llm = LLMFactory.create_llm(llm_config.to_dict())
        except ValueError as e:
            logger.error(f"Failed to create LLM: {e}")
            # Fallback to a basic OpenAI instance if factory fails
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                logger.info("Falling back to basic OpenAI LLM")
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    openai_api_key=api_key,
                    temperature=0.3,
                )
            else:
                raise ValueError("No valid LLM configuration found and no fallback API key available")
        
        logger.info("Initialized SummarizationModule")
    
    def summarize_transcript(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Summarize a meeting transcript and extract potential ideas.
        
        Args:
            transcript: Dictionary containing the transcript with timestamps
            
        Returns:
            List of ideas extracted from the transcript
        """
        logger.info("Summarizing transcript")
        
        # In a real implementation, you would use the LLM to summarize
        # For example:
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", """
        #     You are an expert at analyzing meeting transcripts and extracting potential ideas.
        #     Extract all potential product, feature, or business ideas mentioned in the transcript.
        #     For each idea, provide:
        #     - A concise title
        #     - A detailed description
        #     - A confidence score (0-1) indicating how promising the idea seems
        #     - Relevant timestamps where the idea was discussed
        #     """),
        #     ("human", "{transcript}"),
        # ])
        # 
        # response = self.llm.invoke(prompt.format(transcript=json.dumps(transcript)))
        # ideas = json.loads(response.content)
        
        # For this skeleton, we'll return dummy ideas
        dummy_ideas = [
            {
                "id": str(uuid.uuid4()),
                "title": "AI-Powered Customer Service Assistant",
                "description": "An intelligent assistant that can handle customer inquiries using natural language processing. It would understand customer questions, provide relevant answers, and escalate complex issues to human agents.",
                "confidence_score": 0.9,
                "relevant_timestamps": [
                    {"start": 10.5, "end": 20.0, "speaker": "Speaker 2"},
                    {"start": 21.0, "end": 30.0, "speaker": "Speaker 1"}
                ]
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Smart Product Recommendation Engine",
                "description": "A system that analyzes customer behavior to recommend products they might be interested in. This would increase sales and improve customer satisfaction.",
                "confidence_score": 0.7,
                "relevant_timestamps": [
                    {"start": 35.0, "end": 45.0, "speaker": "Speaker 3"}
                ]
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Automated Meeting Summarizer",
                "description": "A tool that automatically summarizes meetings and extracts action items. This would save time and ensure that important points are not missed.",
                "confidence_score": 0.8,
                "relevant_timestamps": [
                    {"start": 50.0, "end": 60.0, "speaker": "Speaker 1"},
                    {"start": 62.0, "end": 70.0, "speaker": "Speaker 2"}
                ]
            }
        ]
        
        return dummy_ideas
    
    def rank_ideas(self, ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank ideas by their potential.
        
        Args:
            ideas: List of ideas to rank
            
        Returns:
            The same list of ideas, sorted by their potential
        """
        # Sort ideas by confidence score in descending order
        sorted_ideas = sorted(ideas, key=lambda x: x["confidence_score"], reverse=True)
        return sorted_ideas
