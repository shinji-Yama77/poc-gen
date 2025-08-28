"""
MCP (Model Context Protocol) server implementation.
This server orchestrates the workflow for the POC-Gen agent.
"""
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import logging
from typing import Optional, List, Dict, Any

from src.agents.orchestrator import OrchestratorAgent
from src.modules.video_processor import VideoProcessor
from src.modules.transcription import TranscriptionModule
from src.modules.summarization import SummarizationModule
from src.modules.poc_generator import POCGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp_server")

# Create FastAPI app
app = FastAPI(
    title="POC-Gen MCP Server",
    description="MCP server that orchestrates the POC generation process from meeting recordings",
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

class POCGenerationResult(BaseModel):
    """Model for POC generation results."""
    job_id: str
    ideas: List[IdeaSummary]
    selected_ideas: List[str]
    poc_details: Dict[str, Any]
    poc_url: Optional[str] = None

# In-memory storage for job status
job_status = {}

# Initialize components
video_processor = VideoProcessor()
transcription_module = TranscriptionModule()
summarization_module = SummarizationModule()
poc_generator = POCGenerator()
orchestrator = OrchestratorAgent(
    video_processor=video_processor,
    transcription_module=transcription_module,
    summarization_module=summarization_module,
    poc_generator=poc_generator,
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "POC-Gen MCP Server is running"}

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

@app.get("/result/{job_id}", response_model=POCGenerationResult)
async def get_job_result(job_id: str):
    """Get the results of a completed job."""
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
