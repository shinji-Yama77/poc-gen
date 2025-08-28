"""
Orchestrator Agent using LangChain.
This agent coordinates the entire workflow of processing meeting videos,
transcribing them, summarizing into ideas, and generating POCs.
"""
import logging
from typing import Callable, Dict, List, Any, Optional
import tempfile
import os
from fastapi import UploadFile

from langchain.agents import AgentExecutor
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate

from src.modules.video_processor import VideoProcessor
from src.modules.transcription import TranscriptionModule
from src.modules.summarization import SummarizationModule
from src.modules.poc_generator import POCGenerator

logger = logging.getLogger("orchestrator_agent")

class OrchestratorAgent:
    """
    Agent that orchestrates the workflow using LangChain.
    """
    
    def __init__(
        self,
        video_processor: VideoProcessor,
        transcription_module: TranscriptionModule,
        summarization_module: SummarizationModule,
        poc_generator: POCGenerator,
    ):
        """Initialize the orchestrator agent with necessary components."""
        self.video_processor = video_processor
        self.transcription_module = transcription_module
        self.summarization_module = summarization_module
        self.poc_generator = poc_generator
        
        # Initialize the LangChain agent
        self.tools = self._create_tools()
        self.llm = ChatOpenAI(model="gpt-4")
        self.agent_executor = self._create_agent()
    
    def _create_tools(self) -> List[BaseTool]:
        """Create LangChain tools for the agent."""
        
        # Tool for processing video
        class VideoProcessingTool(BaseTool):
            name = "video_processor"
            description = "Process a meeting recording video file"
            
            def __init__(self, video_processor: VideoProcessor):
                super().__init__()
                self.video_processor = video_processor
            
            def _run(self, video_path: str) -> str:
                """Process a video and return the path to processed video."""
                return self.video_processor.process_video(video_path)
        
        # Tool for transcribing video
        class TranscriptionTool(BaseTool):
            name = "transcribe_video"
            description = "Transcribe a processed meeting recording"
            
            def __init__(self, transcription_module: TranscriptionModule):
                super().__init__()
                self.transcription_module = transcription_module
            
            def _run(self, video_path: str) -> Dict[str, Any]:
                """Transcribe a video and return the transcript."""
                return self.transcription_module.transcribe(video_path)
        
        # Tool for summarizing transcript
        class SummarizationTool(BaseTool):
            name = "summarize_transcript"
            description = "Summarize a meeting transcript into potential ideas"
            
            def __init__(self, summarization_module: SummarizationModule):
                super().__init__()
                self.summarization_module = summarization_module
            
            def _run(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
                """Summarize a transcript and return potential ideas."""
                return self.summarization_module.summarize_transcript(transcript)
        
        # Tool for generating POC
        class POCGenerationTool(BaseTool):
            name = "generate_poc"
            description = "Generate a POC from selected ideas"
            
            def __init__(self, poc_generator: POCGenerator):
                super().__init__()
                self.poc_generator = poc_generator
            
            def _run(self, selected_ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
                """Generate a POC from selected ideas."""
                return self.poc_generator.generate_poc(selected_ideas)
        
        # Create and return the tools
        return [
            VideoProcessingTool(self.video_processor),
            TranscriptionTool(self.transcription_module),
            SummarizationTool(self.summarization_module),
            POCGenerationTool(self.poc_generator),
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent executor."""
        
        # Create a prompt for the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI orchestrator agent responsible for processing meeting recordings,
            extracting ideas, and generating POCs. Follow these steps:
            
            1. Process the uploaded meeting video to prepare it for transcription
            2. Transcribe the meeting recording using an open-source AI transcription model
            3. Summarize the transcript into potential ideas
            4. Select the most promising ideas
            5. Generate a POC based on the selected ideas
            
            Use the provided tools to accomplish each step in the workflow.
            """),
            ("human", "{input}"),
        ])
        
        # Create the agent
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        # Create and return the agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
        )
    
    async def process_meeting_recording(
        self,
        job_id: str,
        video_file: UploadFile,
        status_callback: Callable[[str, str, float, str, Optional[Dict[str, Any]]], None],
    ):
        """
        Process a meeting recording through the entire workflow.
        
        Args:
            job_id: Unique identifier for the job
            video_file: Uploaded meeting video file
            status_callback: Callback to update processing status
        """
        logger.info(f"Starting processing job {job_id}")
        
        try:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                # Write the file content
                content = await video_file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Update status
            await status_callback(
                job_id, "processing", 0.1, "Video uploaded, starting processing"
            )
            
            # Process the meeting recording using the LangChain agent
            result = await self._run_agent_workflow(job_id, temp_file_path, status_callback)
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            # Update final status with the result
            await status_callback(
                job_id, "completed", 1.0, "Processing completed", result
            )
            
            logger.info(f"Completed processing job {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            await status_callback(
                job_id, "failed", 0.0, f"Error: {str(e)}"
            )
            # Clean up the temporary file if it exists
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
    
    async def _run_agent_workflow(
        self,
        job_id: str,
        video_path: str,
        status_callback: Callable[[str, str, float, str, Optional[Dict[str, Any]]], None],
    ) -> Dict[str, Any]:
        """
        Run the agent workflow to process the meeting recording.
        
        This method orchestrates the entire workflow using the LangChain agent.
        It handles updating the status at each step of the process.
        """
        # Step 1: Process video
        await status_callback(
            job_id, "processing", 0.2, "Processing video"
        )
        processed_video_path = self.video_processor.process_video(video_path)
        
        # Step 2: Transcribe video
        await status_callback(
            job_id, "processing", 0.4, "Transcribing meeting"
        )
        transcript = self.transcription_module.transcribe(processed_video_path)
        
        # Step 3: Summarize transcript
        await status_callback(
            job_id, "processing", 0.6, "Summarizing transcript and extracting ideas"
        )
        ideas = self.summarization_module.summarize_transcript(transcript)
        
        # Step 4: Select most promising ideas
        await status_callback(
            job_id, "processing", 0.8, "Selecting most promising ideas"
        )
        # This can be done with LangChain agent to select ideas
        selected_ideas = self._select_ideas(ideas)
        
        # Step 5: Generate POC
        await status_callback(
            job_id, "processing", 0.9, "Generating POC"
        )
        poc_result = self.poc_generator.generate_poc(selected_ideas)
        
        # Prepare the final result
        result = {
            "job_id": job_id,
            "ideas": ideas,
            "selected_ideas": [idea["id"] for idea in selected_ideas],
            "poc_details": poc_result,
        }
        
        return result
    
    def _select_ideas(self, ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select the most promising ideas using LLM.
        
        In a real implementation, this would use the LLM to evaluate and select
        the most promising ideas based on criteria like feasibility, innovation, etc.
        """
        # Sort ideas by confidence score and select top 3
        sorted_ideas = sorted(ideas, key=lambda x: x["confidence_score"], reverse=True)
        return sorted_ideas[:3]
