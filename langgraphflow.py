from typing import Literal, Annotated
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from prompts import POC_EXTRACTION_SYSTEM_PROMPT
from pydantic import BaseModel, Field
from typing import List
from langchain.schema import AIMessage

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langgraph.graph import MessagesState

from langchain.chat_models import init_chat_model


# Schema for a single POC
class POCIdea(BaseModel):
    title: str = Field(..., description="Short name for the POC idea")
    problem: str = Field(..., description="Problem this POC is solving")
    solution: str = Field(..., description="Proposed solution to the problem")
    key_features: List[str] = Field(..., description="List of main features")
    tech_stack: List[str] = Field(..., description="Technologies required")
    success_metrics: List[str] = Field(..., description="How success will be measured")
    priority: str = Field(..., description="Priority level, e.g., High, Medium, Low")
    timeline: str = Field(..., description="Expected implementation timeline")

class POCIdeas(BaseModel):
    ideas: List[POCIdea]

class StateInput(TypedDict):
    # this is the input to the state
    transcription_input: dict


class State(MessagesState):
    # This state classes has the messages built in 
    transcription_input: dict
    poc_ideas: POCIdeas = Field(..., description="Structured POC ideas extracted from transcript")


llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_version="2024-12-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Nodes
def extract_poc_ideas(state: State):
    """LLM will extract poc ideas from here"""

    response = llm.with_structured_output(POCIdeas).invoke([
        {"role": "system", "content": POC_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": state["transcription_input"]["content"]}
    ])

    # Wrap the structured JSON as an AIMessage so it can be appended to history
    ai_message = AIMessage(content=response.model_dump_json(indent=2))

    return {
        "messages": [ai_message], # keep conversation history, auto appended by messagesstate
        "poc_ideas": response # store extracted ideas
    }


agent_builder = StateGraph(State, input_schema=StateInput)

# add nodes
agent_builder.add_node("extract_poc", extract_poc_ideas)

# add edges to connect nodes
agent_builder.add_edge(START, "extract_poc")

# Compile the agent
agent = agent_builder.compile()



if __name__ == "__main__":
    # Example transcript input (can be anything, even just plain text)
    input_state = {
        "transcription_input": {
            "content": open("my_transcript.txt").read()
        }
    }

    # Run the agent
    result = agent.invoke(input_state)

    # Print structured results
    print("\n--- Extracted POC Ideas (Structured) ---")
    print(result["poc_ideas"].model_dump_json(indent=2))

    # Print messages (history)
    print("\n--- Conversation History ---")
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")

    png_bytes = agent.get_graph().draw_mermaid_png()
    out_path = "graph.png"

    with open(out_path, "wb") as f:
        f.write(png_bytes)


