"""
Marketability Analysis Module.
Analyzes ideas extracted from meeting transcripts for market potential and audience targeting.
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

logger = logging.getLogger("marketability_analyzer")

class MarketabilityAnalyzerModule:
    """
    Module for analyzing the marketability and target audience for ideas.
    Evaluates audience segments, promotional strategies, and market demand.
    """

    def __init__(self, config: Optional[dict] = None, llm: Optional[BaseChatModel] = None):
        """
        Initialize the marketability analyzer module with configuration.

        Args:
            config: Optional configuration for the analyzer
            llm: Optional pre-configured LLM instance (provided by orchestrator)
        """
        self.config = config or {}

        # Use provided LLM if available
        if llm:
            self.llm = llm
            logger.info("Using externally provided LLM instance for marketability analysis")
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

        logger.info("Initialized MarketabilityAnalyzerModule")

    def analyze_marketability(self, ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze the marketability and target audience for extracted ideas.

        Args:
            ideas: List of ideas extracted from the transcript

        Returns:
            List of ideas with added marketability analysis
        """
        logger.info("Analyzing marketability for %s ideas", len(ideas))

        analyzed_ideas = []

        for idea in ideas:
            try:
                logger.info("Analyzing idea: %s", idea.get('title', 'Untitled'))

                # Create the analysis prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """
                    You are an expert marketing strategist specializing in audience analysis and go-to-market strategies.
                    Analyze the following product/service idea and provide detailed insights on:

                    1. Target Audience: Identify the primary audience segments that would benefit most from this product.
                       Include demographics, psychographics, and behavior patterns.

                    2. Promotional Channels: Recommend specific marketing channels and platforms where this
                       product should focus its promotional efforts for maximum reach and conversion.

                    3. Competitor Marketing Analysis: Analyze where major competitors are focusing their 
                       marketing efforts and identify any underserved audiences or channels.

                    4. Niche Opportunities: Identify any specific population segments that competitors
                       are failing to serve adequately that this product could target.

                    5. Marketing Positioning: Suggest key messaging and positioning strategies to differentiate
                       this product in the marketplace.

                    Format your response as a structured JSON with the following keys:
                    "target_audience", "promotional_channels", "competitor_marketing", "niche_opportunities", "marketing_positioning", "raw_analysis"

                    For target_audience, return an array of objects with "segment", "demographics", "psychographics", and "behaviors" properties.
                    For promotional_channels, return an array of objects with "channel" and "rationale" properties.
                    For competitor_marketing, return an object with "major_competitor_channels" (array) and "insights" (string) properties.
                    For niche_opportunities, return an array of objects with "segment", "opportunity", and "competitor_gaps" properties.
                    For marketing_positioning, return an object with "key_messages" (array), "positioning_strategy", and "unique_value_proposition" properties.

                    Your analysis should be practical, actionable, and grounded in current marketing trends.
                    """),
                    ("human", "{idea_description}"),
                ])

                # Invoke the LLM
                response = self.llm.invoke(prompt.format(idea_description=idea["description"]))

                # Try to extract JSON if the response is not already in JSON format
                content = response.content
                try:
                    # First, try to parse the entire content as JSON
                    marketability_analysis = json.loads(content)
                except json.JSONDecodeError:
                    # If that fails, try to extract JSON using regex
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                    if json_match:
                        try:
                            marketability_analysis = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            # If JSON extraction fails, fall back to parsing manually
                            marketability_analysis = {
                                "target_audience": self._extract_target_audience(content),
                                "promotional_channels": self._extract_promotional_channels(content),
                                "competitor_marketing": self._extract_competitor_marketing(content),
                                "niche_opportunities": self._extract_niche_opportunities(content),
                                "marketing_positioning": self._extract_marketing_positioning(content),
                                "raw_analysis": content
                            }
                    else:
                        # Fall back to manual parsing
                        marketability_analysis = {
                            "target_audience": self._extract_target_audience(content),
                            "promotional_channels": self._extract_promotional_channels(content),
                            "competitor_marketing": self._extract_competitor_marketing(content),
                            "niche_opportunities": self._extract_niche_opportunities(content),
                            "marketing_positioning": self._extract_marketing_positioning(content),
                            "raw_analysis": content
                        }

                # Add the analysis to the idea
                idea_with_analysis = idea.copy()
                idea_with_analysis["marketability_analysis"] = marketability_analysis
                analyzed_ideas.append(idea_with_analysis)

            except (ValueError, TypeError, json.JSONDecodeError) as e:
                logger.error("Error analyzing idea %s: %s", idea.get('id', 'unknown'), str(e))
                # Add the idea with an error message
                idea_with_error = idea.copy()
                idea_with_error["marketability_analysis"] = {
                    "error": str(e),
                    "raw_analysis": "Analysis failed due to an error."
                }
                analyzed_ideas.append(idea_with_error)

        return analyzed_ideas

    def _extract_target_audience(self, analysis_text: str) -> List[Dict[str, Any]]:
        """Extract the target audience segments from the analysis text."""
        # todo implement later
        if not analysis_text:
            return []
        return [{
            "segment": "Primary Audience Segment",
            "demographics": "Key demographics",
            "psychographics": "Key psychographics",
            "behaviors": "Key behaviors"
        }]

    def _extract_promotional_channels(self, analysis_text: str) -> List[Dict[str, Any]]:
        """Extract recommended promotional channels from the analysis text."""
        # todo implement later
        if not analysis_text:
            return []
        return [{
            "channel": "Channel name",
            "rationale": "Why this channel is recommended"
        }]

    def _extract_competitor_marketing(self, analysis_text: str) -> Dict[str, Any]:
        """Extract competitor marketing analysis from the analysis text."""
        # todo implement later
        if not analysis_text:
            return {"major_competitor_channels": [], "insights": "No insights available"}
        return {
            "major_competitor_channels": ["Channel 1", "Channel 2"],
            "insights": "Insights about competitor marketing strategies"
        }

    def _extract_niche_opportunities(self, analysis_text: str) -> List[Dict[str, Any]]:
        """Extract niche opportunities from the analysis text."""
        # todo implement later
        if not analysis_text:
            return []
        return [{
            "segment": "Underserved segment",
            "opportunity": "Description of the opportunity",
            "competitor_gaps": "How competitors are missing this segment"
        }]

    def _extract_marketing_positioning(self, analysis_text: str) -> Dict[str, Any]:
        """Extract marketing positioning recommendations from the analysis text."""
        # todo implement later
        if not analysis_text:
            return {
                "key_messages": [],
                "positioning_strategy": "No positioning strategy available", 
                "unique_value_proposition": "No unique value proposition available"
            }
        return {
            "key_messages": ["Message 1", "Message 2"],
            "positioning_strategy": "Recommended positioning strategy",
            "unique_value_proposition": "The unique value proposition"
        }
