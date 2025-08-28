# POC-Gen: Meeting Recording to POC Generator

An AI agent that converts meeting recordings into proof-of-concept (POC) applications.

## Project Overview

POC-Gen automates the process of turning ideas from meeting recordings into functional proof-of-concept applications. It works through the following steps:

1. **Video Processing**: Takes a user's video recording from a meeting
2. **Transcription**: Transcribes the meeting using open-source AI
3. **Idea Extraction**: Summarizes the meeting contents into potential ideas
4. **Idea Selection**: Selects the most promising ideas
5. **POC Generation**: Turns selected ideas into a prompt for a no-code POC builder AI (like Alchemist AI)
6. **Delivery**: Returns the POC to the user

## Project Structure

```
poc-gen/
├── src/
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   └── server.py             # MCP server implementation
│   ├── agents/
│   │   ├── __init__.py
│   │   └── orchestrator.py       # LangChain orchestrator agent
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── video_processor.py    # Video processing module
│   │   ├── transcription.py      # Transcription module
│   │   ├── summarization.py      # Meeting summarization module
│   │   └── poc_generator.py      # POC generation module
│   └── __init__.py
├── main.py                       # Entry point
├── pyproject.toml                # Project dependencies
├── README.md                     # Project documentation
└── uv.lock                       # Dependency lock file
```

## Key Components

### MCP Server (Model Context Protocol)

The MCP server acts as the orchestrator for all AI communications. It exposes APIs for:
- Uploading meeting recordings
- Checking processing status
- Retrieving results

### LangChain Agent Framework

The project uses LangChain to create a structured agent workflow:
- Coordinates multiple AI models
- Manages the processing pipeline
- Handles complex reasoning tasks

### Functional Modules

- **Video Processor**: Prepares video for transcription
- **Transcription**: Uses open-source AI to convert speech to text
- **Summarization**: Extracts and analyzes ideas from the transcript
- **POC Generator**: Creates prompts for the no-code POC builder

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd poc-gen
   ```

2. Install dependencies:
   ```bash
   uv pip install -e .
   ```

3. Set up environment variables:
   ```bash
   # For development
   export OPENAI_API_KEY=your_openai_api_key
   export POC_BUILDER_API_KEY=your_poc_builder_api_key  # For Alchemist AI or similar
   ```

### Running the Application

1. Start the MCP server:
   ```bash
   python main.py
   ```

2. Access the API at `http://localhost:8000`

## API Endpoints

- `POST /upload-meeting`: Upload a meeting recording
- `GET /status/{job_id}`: Check processing status
- `GET /result/{job_id}`: Get processing results

## Future Enhancements

- Add authentication and user management
- Implement real-time processing status updates
- Add support for different POC builder platforms
- Create a web interface for easier interaction
- Add support for more meeting recording formats
- Implement feedback loop for improving idea selection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
