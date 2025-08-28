"""
POC-Gen: AI agent that converts meeting recordings into POC applications.
Main entry point for the application.
"""
import uvicorn
import os
from dotenv import load_dotenv
from src.mcp_server.server import app

def main():
    """
    Start the MCP server which orchestrates the entire workflow:
    1. Receive meeting video
    2. Transcribe the video
    3. Summarize the meeting and extract ideas
    4. Generate POCs from selected ideas
    5. Return POC to the user
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables")
    
    if not os.environ.get("POC_BUILDER_API_KEY"):
        print("Warning: POC_BUILDER_API_KEY not found in environment variables")
    
    print("Starting POC-Gen AI Agent...")
    # Run the FastAPI MCP server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
