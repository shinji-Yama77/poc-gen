# POC-Gen: Pitch Recording to AI Business Report Generator

An AI agent that converts pitch recordings into actionable business reports focused on monetization and marketing strategies.

## Project Overview

POC-Gen automates the process of turning user pitch recordings into comprehensive business reports. It works through the following steps:

1. **Audio/Video Upload**: Takes a user's pitch recording (audio from video)
2. **Transcription & Structuring**: Transcribes the pitch using open-source AI and structures output as JSON for downstream processing
3. **Idea Extraction**: Summarizes the transcript into 2–3 key product ideas
4. **Report Generation**: For each idea, generates a detailed business report emphasizing:
    - Monetization strategies (including competitor analysis and revenue projections)
    - Marketing strategies (including differentiation and go-to-market planning)
    - Success metrics, standout factors, and actionable insights
5. **Delivery**: Returns the business reports to the user

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
│   │   ├── audio_processor.py    # Audio/video processing module
│   │   ├── transcription.py      # Transcription module
│   │   ├── summarization.py      # Idea extraction/summarization module
│   │   ├── report_generator.py   # Business report generation module
│   │   └── competitor_analysis.py # Competitor data gathering & analysis
│   └── __init__.py
├── main.py                       # Entry point
├── pyproject.toml                # Project dependencies
├── README.md                     # Project documentation
└── uv.lock                       # Dependency lock file
```

## Key Components

### MCP Server (Model Context Protocol)

Acts as the orchestrator for all AI communications. Exposes APIs for:
- Uploading pitch recordings
- Checking processing status
- Retrieving business reports

### LangChain Agent Framework

Uses LangChain to create a structured agent workflow:
- Coordinates multiple AI models
- Manages the processing pipeline
- Handles reasoning and report generation

### Functional Modules

- **Audio Processor**: Prepares audio/video for transcription
- **Transcription**: Uses open-source AI to transcribe speech to text
- **Summarization**: Extracts and analyzes ideas from the transcript
- **Report Generator**: Creates business reports focused on monetization and marketing
- **Competitor Analysis**: Gathers and analyzes competitor data for benchmarking

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
   export COMPETITOR_API_KEY=your_competitor_data_api_key  # If needed
   ```

### Running the Application

1. Start the MCP server:
   ```bash
   python main.py
   ```

2. Access the API at `http://localhost:8000`

## API Endpoints

- `POST /upload-pitch`: Upload a pitch recording
- `GET /status/{job_id}`: Check processing status
- `GET /result/{job_id}`: Get generated business reports

## Output Example

Each idea receives a structured business report, including:
- Monetization strategies and competitor benchmarks
- Estimated revenue and success analysis
- Marketing playbook with differentiation tactics
- Visuals (e.g., Mermaid diagrams for business/process flows)
- Actionable next steps

## Future Enhancements

- Add authentication and user management
- Real-time processing status updates
- Advanced competitor intelligence via web search
- Web dashboard for interactive report viewing
- Multi-format report downloads (Markdown, PDF, PPT)
- Feedback loop for improving report quality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)