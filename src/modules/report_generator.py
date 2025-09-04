"""
Report Generation Module.
Generates comprehensive reports from the analysis of ideas.
"""
import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

from src.core.llm_factory import LLMFactory, LLMConfig

logger = logging.getLogger("report_generator")

class ReportGeneratorModule:
    """
    Module for generating comprehensive reports from analyzed ideas.
    Compiles monetization and marketability analyses into a structured report.
    """
    
    def __init__(self, config: Optional[dict] = None, llm: Optional[BaseChatModel] = None):
        """
        Initialize the report generator module with configuration.
        
        Args:
            config: Optional configuration for the generator
            llm: Optional pre-configured LLM instance (provided by orchestrator)
        """
        self.config = config or {}
        
        # Use provided LLM if available
        if llm:
            self.llm = llm
            logger.info("Using externally provided LLM instance for report generation")
            return
        
        # Otherwise, create own LLM instance using factory (for backward compatibility)
        logger.info("Creating new LLM instance for report generation")
        
        # Convert config to LLMConfig if needed
        if isinstance(config, dict):
            llm_config = LLMConfig(
                ai_provider=config.get("ai_provider", "openai"),
                temperature=config.get("temperature", 0.3),
                api_key=config.get("api_key"),
                model=config.get("model", "gpt-4o"),
                azure_api_key=config.get("azure_api_key"),
                azure_endpoint=config.get("azure_endpoint"),
                azure_deployment=config.get("azure_deployment"),
                azure_api_version=config.get("azure_api_version"),
            )
        else:
            llm_config = LLMConfig.from_env()
        
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
                    model="gpt-4o",
                    openai_api_key=api_key,
                    temperature=0.3,
                )
            else:
                raise ValueError("No valid LLM configuration found and no fallback API key available")
        
        logger.info("Initialized ReportGeneratorModule")
    
    def generate_report(self, analyzed_ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive report from analyzed ideas.
        
        Args:
            analyzed_ideas: List of ideas with monetization and marketability analyses
            
        Returns:
            A structured report with executive summary, detailed analyses, and recommendations
        """
        logger.info(f"Generating report for {len(analyzed_ideas)} analyzed ideas")
        
        # Generate an executive summary
        executive_summary = self._generate_executive_summary(analyzed_ideas)
        
        # Generate detailed analysis for each idea
        idea_reports = []
        for idea in analyzed_ideas:
            idea_report = self._generate_idea_report(idea)
            idea_reports.append(idea_report)
        
        # Generate overall recommendations
        recommendations = self._generate_recommendations(analyzed_ideas)
        
        # Compile the full report
        report = {
            "report_id": f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": executive_summary,
            "idea_reports": idea_reports,
            "recommendations": recommendations
        }
        
        return report
    
    def _generate_executive_summary(self, analyzed_ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an executive summary for the analyzed ideas."""
        
        # Create a prompt for the executive summary
        ideas_summary = "\n\n".join([
            f"Idea: {idea['title']}\nDescription: {idea['description']}" 
            for idea in analyzed_ideas
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert business analyst creating an executive summary for a set of business ideas.
            Your task is to create a concise, insightful executive summary that highlights:
            
            1. The overall potential of the ideas presented
            2. Key market opportunities identified
            3. Major challenges or concerns across the ideas
            4. A high-level assessment of which ideas show the most promise
            
            Keep your summary professional, data-driven, and actionable. Limit to 3-5 paragraphs.
            """),
            ("human", f"Here are the ideas to summarize:\n\n{ideas_summary}"),
        ])
        
        # Invoke the LLM
        response = self.llm.invoke(prompt)
        
        return {
            "summary": response.content,
            "top_ideas": self._identify_top_ideas(analyzed_ideas),
            "overall_feasibility": self._calculate_overall_feasibility(analyzed_ideas)
        }
    
    def _generate_idea_report(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a detailed report for a single idea."""
        
        # Extract the relevant analyses
        monetization = idea.get("monetization_analysis", {})
        marketability = idea.get("marketability_analysis", {})
        
        # Generate a synthesis of the analyses
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert business consultant creating a concise synthesis of monetization 
            and marketability analyses for a business idea. Create a cohesive, insightful synthesis 
            that highlights the connections between market potential and monetization strategy.
            Focus on actionable insights and strategic recommendations.
            """),
            ("human", f"""
            Idea: {idea['title']}
            
            Description: {idea['description']}
            
            Monetization Analysis: {monetization.get('raw_analysis', 'No monetization analysis available.')}
            
            Marketability Analysis: {marketability.get('raw_analysis', 'No marketability analysis available.')}
            """),
        ])
        
        # Invoke the LLM for synthesis
        synthesis_response = self.llm.invoke(synthesis_prompt)
        
        # Compile the idea report
        return {
            "idea_id": idea.get("id", "unknown"),
            "title": idea.get("title", "Untitled Idea"),
            "description": idea.get("description", "No description available"),
            "confidence_score": idea.get("confidence_score", 0),
            "monetization_analysis": monetization,
            "marketability_analysis": marketability,
            "synthesis": synthesis_response.content,
            "relevant_timestamps": idea.get("relevant_timestamps", [])
        }
    
    def _generate_recommendations(self, analyzed_ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall recommendations based on all analyzed ideas."""
        
        # Create a prompt for recommendations
        ideas_summary = "\n\n".join([
            f"Idea: {idea['title']}\nDescription: {idea['description']}\n" +
            f"Monetization: {idea.get('monetization_analysis', {}).get('raw_analysis', 'N/A')}\n" +
            f"Marketability: {idea.get('marketability_analysis', {}).get('raw_analysis', 'N/A')}"
            for idea in analyzed_ideas
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert business strategist providing actionable recommendations based on 
            analyzed business ideas. Create specific, practical recommendations in these categories:
            
            1. Next Steps: What immediate actions should be taken to validate or develop these ideas?
            2. Resource Allocation: How should resources be prioritized across these ideas?
            3. Market Entry Strategy: What approach should be taken to enter the market?
            4. Risk Mitigation: What are the key risks and how should they be addressed?
            5. Timeline: What is a realistic timeline for developing and launching these ideas?
            
            Your recommendations should be specific, actionable, and grounded in the analyses provided.
            """),
            ("human", f"Here are the analyzed ideas:\n\n{ideas_summary}"),
        ])
        
        # Invoke the LLM
        response = self.llm.invoke(prompt)
        
        # Structure the recommendations
        return {
            "strategic_recommendations": response.content,
            "prioritized_ideas": self._prioritize_ideas(analyzed_ideas)
        }
    
    def _identify_top_ideas(self, analyzed_ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify the top ideas based on combined monetization and marketability potential."""
        # In a real implementation, you would have a more sophisticated ranking algorithm
        # This is a simplified version
        
        # Sort ideas by confidence score
        sorted_ideas = sorted(analyzed_ideas, key=lambda x: x.get("confidence_score", 0), reverse=True)
        
        # Return the top 3 or fewer
        top_ideas = sorted_ideas[:min(3, len(sorted_ideas))]
        
        return [{"id": idea.get("id", "unknown"), "title": idea.get("title", "Untitled")} for idea in top_ideas]
    
    def _calculate_overall_feasibility(self, analyzed_ideas: List[Dict[str, Any]]) -> float:
        """Calculate the overall feasibility score across all ideas."""
        # In a real implementation, you would have a more sophisticated calculation
        # This is a simplified version
        
        feasibility_scores = [
            idea.get("monetization_analysis", {}).get("feasibility", {}).get("score", 5)
            for idea in analyzed_ideas
            if "monetization_analysis" in idea
        ]
        
        if not feasibility_scores:
            return 5.0  # Default mid-point
        
        return sum(feasibility_scores) / len(feasibility_scores)
    
    def _prioritize_ideas(self, analyzed_ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize ideas based on combined analysis factors."""
        # This would be a more complex algorithm in a real implementation
        
        # For now, just sort by confidence score
        sorted_ideas = sorted(analyzed_ideas, key=lambda x: x.get("confidence_score", 0), reverse=True)
        
        return [
            {
                "id": idea.get("id", "unknown"),
                "title": idea.get("title", "Untitled"),
                "priority_score": idea.get("confidence_score", 0),
                "rationale": f"Based on confidence score of {idea.get('confidence_score', 0)}"
            }
            for idea in sorted_ideas
        ]
    
    def save_report(self, report: Dict[str, Any], output_dir: str) -> str:
        """
        Save the generated report to a file.
        
        Args:
            report: The report to save
            output_dir: Directory to save the report in
            
        Returns:
            Path to the saved report file
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a filename for the report
        filename = f"idea_analysis_report_{report['report_id']}.json"
        output_path = os.path.join(output_dir, filename)
        
        # Save the report to the output path
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved report to: {output_path}")
        return output_path
