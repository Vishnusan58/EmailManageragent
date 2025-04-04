from mcp.server.fastmcp import FastMCP
from pinecone import Pinecone
from openai import OpenAI as OpenAIClient

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key="apikey")  # Replace with your Pinecone API key
index = pc.Index("work-mail")
openai_client = OpenAIClient(api_key="apikey")  # Replace with your OpenAI API key

# Create an MCP server
mcp = FastMCP("EmailTools")

# Define MCP Tools
@mcp.tool()
def search_emails(query: str) -> list:
    """Search for related emails by keyword."""
    embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding
    results = index.query(vector=embedding, top_k=2, include_metadata=True)["matches"]
    return [{"subject": r["metadata"]["subject"], "content": r["metadata"]["content"][:200]} for r in results]

@mcp.tool()
def get_calendar_events(days: int) -> dict:
    """Get upcoming calendar events for the next N days."""
    # Mock calendar data (replace with real API like Google Calendar)
    return {"events": ["Meeting at 10 AM Wed", "Call at 3 PM Thu"]}

if __name__ == "__main__":
    # Run the MCP server (assumes it defaults to localhost:8000)
    mcp.run(host="0.0.0.0", port=8000)