# %%

from pathlib import Path

import click
import gradio as gr

from unfccc_rag.retrieval import load_retrieval_engine


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
def launch(embeddings_file: Path, model_name: str):
    rag_query = load_retrieval_engine(embeddings_file, model_name)

    def rag_query_wrapper(message: str, history):
        return rag_query(message)

    demo = gr.ChatInterface(
        rag_query_wrapper,
        type="messages",
        flagging_mode="manual",
        flagging_options=["Like", "Spam", "Inappropriate", "Other"],
        save_history=True,
        title="🌍 UN Climate Documents RAG System",
        description="Ask questions about UN climate change decisions and get answers with sources",
    )
    demo.launch()
