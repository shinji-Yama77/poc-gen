"""
Test script for the Monetization Analyzer Module using LLMFactory directly.
This script demonstrates how to use the module with the LLMFactory as intended.
"""
import json
import logging
from dotenv import load_dotenv
from src.core.llm_factory import LLMFactory, LLMConfig
from src.modules.monetization_analyzer import MonetizationAnalyzerModule

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

def test_monetization_analyzer():
    """Test the MonetizationAnalyzerModule with sample ideas."""

    # First create the LLM instance using LLMFactory
    llm_config = LLMConfig.from_env()
    llm = LLMFactory.create_llm(llm_config.to_dict())

    # Then pass the LLM instance to MonetizationAnalyzerModule
    analyzer = MonetizationAnalyzerModule(llm=llm)

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
            "title": "Subscription-based plant care service",
            "description": """
            A subscription service that delivers personalized plant care products 
            and instructions based on the specific plants in your home. Users would 
            upload photos of their plants through an app, and the service would identify 
            the plants, assess their health, and send customized care packages including 
            the right soil, fertilizer, and treatment products on a regular schedule. 
            The app would also provide care reminders and troubleshooting advice.
            """
        }
    ]

    # Analyze the ideas
    analyzed_ideas = analyzer.analyze_monetization(sample_ideas)

    # Print the results
    for idea in analyzed_ideas:
        print(f"\n{'='*80}")
        print(f"IDEA: {idea['title']}")
        print(f"{'='*80}")
        print("\nMONETIZATION ANALYSIS:")
        print(json.dumps(idea['monetization_analysis'], indent=2))
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    test_monetization_analyzer()
