"""
Monetization Analysis Module.
Analyzes ideas extracted from meeting transcripts for monetization potential.
"""
import logging
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

from src.core.llm_factory import LLMFactory, LLMConfig

load_dotenv()

logger = logging.getLogger("monetization_analyzer")

class MonetizationAnalyzerModule:
    """
    Module for analyzing the monetization potential of ideas.
    Evaluates market competition, revenue potential, and feasibility.
    """

    def __init__(self, config: Optional[dict] = None, llm: Optional[BaseChatModel] = None):
        """
        Initialize the monetization analyzer module with configuration.

        Args:
            config: Optional configuration for the analyzer
            llm: Optional pre-configured LLM instance (provided by orchestrator)
        """
        self.config = config or {}

        # Use provided LLM if available
        if llm:
            self.llm = llm
            logger.info("Using externally provided LLM instance for monetization analysis")
            return

        # Create LLM using factory
        try:
            # Convert config to LLMConfig if needed
            if isinstance(self.config, dict):
                llm_config = LLMConfig(
                    ai_provider=self.config.get("ai_provider", "azure"),
                    temperature=self.config.get("temperature", 0.3),
                    azure_api_key=self.config.get("azure_api_key"),
                    azure_endpoint=self.config.get("azure_endpoint"),
                    azure_deployment=self.config.get("azure_deployment"),
                    azure_api_version=self.config.get("azure_api_version"),
                )
                self.llm = LLMFactory.create_llm(llm_config.to_dict())
            else:
                # Fallback to environment variables
                llm_config = LLMConfig.from_env()
                self.llm = LLMFactory.create_llm(llm_config.to_dict())
        except ValueError as e:
            logger.error("Failed to create LLM: %s", e)
            raise ValueError(
                "No valid LLM configuration found and no fallback API key available") from e

        logger.info("Initialized MonetizationAnalyzerModule")

    def analyze_monetization(self, ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze the monetization potential of extracted ideas.

        Args:
            ideas: List of ideas extracted from the transcript

        Returns:
            List of ideas with added monetization analysis
        """
        logger.info("Analyzing monetization potential for %s ideas", len(ideas))

        analyzed_ideas = []

        for idea in ideas:
            try:
                logger.info("Analyzing idea: %s", idea.get('title', 'Untitled'))

                # Create the analysis prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """
                    You are an expert business analyst specializing in monetization strategy. 
                    Analyze the following product/service idea and provide detailed insights on:

                    1. Target Industry: Identify the primary industry this product would serve.
                    2. Leading Competitors: Identify 3-5 major competitors in this space and how they differ from this idea.
                    3. Differentiation Strategy: Recommend how this product can set itself apart from existing solutions.
                    4. Areas for Iteration: Suggest specific aspects of the idea that should be improved to increase success probability.
                    5. Feasibility Score: Rate the overall feasibility from 1-10, with justification based on market saturation, entry barriers, and demand.

                    Format your response as a structured JSON with the following keys:
                    "target_industry", "competitors", "differentiation", "iteration_areas", "feasibility", "raw_analysis"

                    For competitors, return an array of objects with "name" and "differentiation" properties.
                    For iteration_areas, return an array of strings.
                    For feasibility, return an object with "score" (integer 1-10) and "justification" (string).

                    Your analysis should be data-driven, practical, and actionable. Be honest about market challenges but constructive in your recommendations.
                    """),
                    ("human", "{idea_description}"),
                ])

                # Invoke the LLM
                response = self.llm.invoke(
                    prompt.format(idea_description=idea["description"]))

                # Try to extract JSON if the response is not already in JSON format
                content = response.content
                try:
                    # First, try to parse the entire content as JSON
                    monetization_analysis = json.loads(content)
                except json.JSONDecodeError:
                    # If that fails, try to extract JSON using regex
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                    if json_match:
                        try:
                            monetization_analysis = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            # If JSON extraction fails, fall back to parsing manually
                            monetization_analysis = {
                                "target_industry": self._extract_target_industry(content),
                                "competitors": self._extract_competitors(content),
                                "differentiation": self._extract_differentiation(content),
                                "iteration_areas": self._extract_iteration_areas(content),
                                "feasibility": self._extract_feasibility(content),
                                "raw_analysis": content
                            }
                    else:
                        # Fall back to manual parsing
                        monetization_analysis = {
                            "target_industry": self._extract_target_industry(content),
                            "competitors": self._extract_competitors(content),
                            "differentiation": self._extract_differentiation(content),
                            "iteration_areas": self._extract_iteration_areas(content),
                            "feasibility": self._extract_feasibility(content),
                            "raw_analysis": content
                        }

                # Add the analysis to the idea
                idea_with_analysis = idea.copy()
                idea_with_analysis["monetization_analysis"] = monetization_analysis
                analyzed_ideas.append(idea_with_analysis)

            except (ValueError, TypeError, json.JSONDecodeError) as e:
                logger.error("Error analyzing idea %s: %s", idea.get('id', 'unknown'), str(e))
                # Add the idea with an error message
                idea_with_error = idea.copy()
                idea_with_error["monetization_analysis"] = {
                    "error": str(e),
                    "raw_analysis": "Analysis failed due to an error."
                }
                analyzed_ideas.append(idea_with_error)

        return analyzed_ideas

    def _extract_target_industry(self, analysis_text: str) -> str:
        """Extract the target industry from the analysis text."""
        # todo implement later
        if not analysis_text:
            return "Unknown industry"
        return "Extracted target industry"

    def _extract_competitors(self, analysis_text: str) -> List[Dict[str, str]]:
        """Extract competitors from the analysis text."""
        # todo implement later
        if not analysis_text:
            return []
        return [{"name": "Competitor 1", "differentiation": "How they differ"}]

    def _extract_differentiation(self, analysis_text: str) -> str:
        """Extract differentiation strategy from the analysis text."""
        # todo implement later
        if not analysis_text:
            return "No differentiation strategy found"
        return "Extracted differentiation strategy"

    def _extract_iteration_areas(self, analysis_text: str) -> List[str]:
        """Extract areas for iteration from the analysis text."""
        # todo implement later
        if not analysis_text:
            return []
        return ["Iteration area 1", "Iteration area 2"]

    def _extract_feasibility(self, analysis_text: str) -> Dict[str, Any]:
        """Extract feasibility score and justification from the analysis text."""
        # todo implement later
        if not analysis_text:
            return {"score": 0, "justification": "Insufficient data for feasibility analysis"}
        return {
            "score": 7,  # Example score
            "justification": "Extracted justification for the feasibility score"
        }
