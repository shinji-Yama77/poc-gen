"""
MCP (Model Context Protocol) server implementation.
This server orchestrates the workflow for analyzing meeting recordings to extract
business ideas, analyze their monetization and marketability potential, and generate
comprehensive reports.
"""
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import logging
from typing import Optional, List, Dict, Any

from src.agents.orchestrator import OrchestratorAgent
from src.core.llm_factory import LLMFactory, LLMConfig
from src.modules.video_processor import VideoProcessor
from src.modules.transcription import TranscriptionModule
from src.modules.summarization import SummarizationModule
from src.modules.monetization_analyzer import MonetizationAnalyzerModule
from src.modules.marketability_analyzer import MarketabilityAnalyzerModule
from src.modules.report_generator import ReportGeneratorModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp_server")

# Create FastAPI app
app = FastAPI(
    title="Business Idea Analyzer MCP Server",
    description="MCP server that analyzes meeting recordings to extract and evaluate business ideas",
    version="0.1.0",
)

# Models for request/response
class ProcessingStatus(BaseModel):
    """Model for tracking processing status."""
    job_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None

class IdeaSummary(BaseModel):
    """Model for representing an idea summary."""
    id: str
    title: str
    description: str
    confidence_score: float
    relevant_timestamps: List[Dict[str, Any]]

class MonetizationAnalysis(BaseModel):
    """Model for monetization analysis."""
    target_industry: str
    competitors: List[Dict[str, Any]]
    differentiation: str
    iteration_areas: List[str]
    feasibility: Dict[str, Any]
    raw_analysis: Optional[str] = None

class MarketabilityAnalysis(BaseModel):
    """Model for marketability analysis."""
    target_audience: List[Dict[str, Any]]
    promotional_channels: List[Dict[str, Any]]
    competitor_marketing: Dict[str, Any]
    niche_opportunities: List[Dict[str, Any]]
    marketing_positioning: Dict[str, Any]
    raw_analysis: Optional[str] = None

class AnalyzedIdea(BaseModel):
    """Model for a fully analyzed idea."""
    id: str
    title: str
    description: str
    confidence_score: float
    relevant_timestamps: List[Dict[str, Any]]
    monetization_analysis: MonetizationAnalysis
    marketability_analysis: MarketabilityAnalysis

class AnalysisReport(BaseModel):
    """Model for the final analysis report."""
    report_id: str
    generated_at: str
    executive_summary: Dict[str, Any]
    idea_reports: List[Dict[str, Any]]
    recommendations: Dict[str, Any]

class AnalysisResult(BaseModel):
    """Model for the complete analysis results."""
    job_id: str
    ideas: List[AnalyzedIdea]
    report: AnalysisReport

# In-memory storage for job status
job_status = {}

# Initialize centralized LLM
logger.info("Initializing centralized LLM for MCP server")
try:
    # Create LLM configuration from environment
    llm_config = LLMConfig.from_env()
    shared_llm = LLMFactory.create_llm(llm_config.to_dict())
    
    # Log which provider is being used
    provider_info = LLMFactory.get_provider_info(llm_config.to_dict())
    logger.info(f"Using AI provider: {provider_info}")
    
except Exception as e:
    logger.error(f"Failed to initialize centralized LLM: {e}")
    logger.warning("Modules will fall back to individual LLM initialization")
    shared_llm = None

# Initialize components with shared LLM
video_processor = VideoProcessor()
transcription_module = TranscriptionModule()
summarization_module = SummarizationModule(llm=shared_llm)
monetization_analyzer = MonetizationAnalyzerModule(llm=shared_llm)
marketability_analyzer = MarketabilityAnalyzerModule(llm=shared_llm)
report_generator = ReportGeneratorModule(llm=shared_llm)

orchestrator = OrchestratorAgent(
    video_processor=video_processor,
    transcription_module=transcription_module,
    summarization_module=summarization_module,
    monetization_analyzer=monetization_analyzer,
    marketability_analyzer=marketability_analyzer,
    report_generator=report_generator,
    shared_llm=shared_llm
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Business Idea Analyzer MCP Server is running"}

@app.post("/upload-meeting", response_model=ProcessingStatus)
async def upload_meeting(
    background_tasks: BackgroundTasks,
    meeting_video: UploadFile = File(...),
):
    """
    Upload a meeting recording to be processed.
    The processing will happen in the background.
    """
    # Generate a unique job ID
    job_id = f"job_{asyncio.create_task(asyncio.sleep(0)).get_name()}"
    
    # Save job status
    job_status[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "message": "Starting video processing",
    }
    
    # Process the video in the background
    background_tasks.add_task(
        orchestrator.process_meeting_recording,
        job_id=job_id,
        video_file=meeting_video,
        status_callback=update_job_status,
    )
    
    return ProcessingStatus(
        job_id=job_id,
        status="processing",
        progress=0.0,
        message="Your meeting video has been uploaded and is being processed",
    )

@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    if job_id not in job_status:
        return JSONResponse(
            status_code=404,
            content={"message": f"Job {job_id} not found"},
        )
    
    status_data = job_status[job_id]
    return ProcessingStatus(
        job_id=job_id,
        status=status_data["status"],
        progress=status_data["progress"],
        message=status_data["message"],
    )

@app.get("/result/{job_id}", response_model=AnalysisResult)
async def get_job_result(job_id: str):
    """Get the analysis results of a completed job."""
    if job_id not in job_status:
        return JSONResponse(
            status_code=404,
            content={"message": f"Job {job_id} not found"},
        )
    
    status_data = job_status[job_id]
    if status_data["status"] != "completed":
        return JSONResponse(
            status_code=400,
            content={"message": f"Job {job_id} is not completed yet"},
        )
    
    return status_data["result"]

# Helper function to update job status
async def update_job_status(job_id: str, status: str, progress: float, message: str, result: Optional[Dict[str, Any]] = None):
    """Update the status of a job."""
    job_status[job_id] = {
        "status": status,
        "progress": progress,
        "message": message,
    }
    
    if result:
        job_status[job_id]["result"] = result
