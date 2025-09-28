"""Start the UNFCCC RAG MCP Server."""

import logging
from pathlib import Path

import click
from mcp.server.fastmcp import FastMCP

from unfccc_rag.retrieval import load_retrieval_engine

logger = logging.getLogger(__name__)


mcp = FastMCP(
    name="rag-mcp",
    instructions="This server provides a tool to retrieve documents from the UNFCCC climate change negotiations.",
)


@click.command()
@click.option(
    "--embeddings-file",
    type=Path,
    required=True,
    help="Path to the embeddings file.",
)
@click.option(
    "--model-name",
    type=str,
    default="google/embeddinggemma-300m",
    help="Name of the model to use for the embeddings.",
)
def start_server(embeddings_file: Path, model_name: str):
    # Get fresh config to ensure we have latest environment variables
    logger.info("Starting UNFCCC RAG MCP Server")
    rag_query = load_retrieval_engine(embeddings_file, model_name)
    mcp.add_tool(rag_query)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    start_server()
