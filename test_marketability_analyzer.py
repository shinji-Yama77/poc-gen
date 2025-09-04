"""
Test script for the Marketability Analyzer Module using LLMFactory directly.
This script demonstrates how to use the module with the LLMFactory as intended.
"""
import json
import logging
from dotenv import load_dotenv
from src.core.llm_factory import LLMFactory, LLMConfig
from src.modules.marketability_analyzer import MarketabilityAnalyzerModule

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

def test_marketability_analyzer():
    """Test the MarketabilityAnalyzerModule with sample ideas."""

    # First create the LLM instance using LLMFactory
    llm_config = LLMConfig.from_env()
    llm = LLMFactory.create_llm(llm_config.to_dict())

    # Then pass the LLM instance to MarketabilityAnalyzerModule
    analyzer = MarketabilityAnalyzerModule(llm=llm)

    # Sample ideas to analyze
    sample_ideas = [
        {
            "id": 1,
            "title": "AI-powered personal shopping assistant",
            "description": """
            An AI-powered personal shopping assistant that learns user preferences 
            over time and helps them find products they'll love across multiple 
            e-commerce platforms. The assistant would analyze past purchases, 
            browsing history, and stated preferences to make personalized 
            recommendations. It would also alert users to price drops on items 
            they're interested in and suggest alternatives when items are out of stock.
            """
        },
        {
            "id": 2,
            "title": "Smart home energy optimization system",
            "description": """
            A smart home system that automatically optimizes energy usage based on 
            resident behavior patterns, weather forecasts, and energy price fluctuations.
            The system would use AI to learn when residents are typically home, adjust 
            heating/cooling systems accordingly, and manage smart appliances to run during
            off-peak hours. It would provide a dashboard showing energy savings and 
            environmental impact, with suggestions for further optimization.
            """
        }
    ]

    # Analyze the ideas
    analyzed_ideas = analyzer.analyze_marketability(sample_ideas)

    # Print the results
    for idea in analyzed_ideas:
        print(f"\n{'='*80}")
        print(f"IDEA: {idea['title']}")
        print(f"{'='*80}")
        print("\nMARKETABILITY ANALYSIS:")
        print(json.dumps(idea['marketability_analysis'], indent=2))
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    test_marketability_analyzer()
