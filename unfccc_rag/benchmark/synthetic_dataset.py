"""Generate document-grounded synthetic questions/answer pairs dataset via the OpenAI Responses API."""

import json
import logging
from pathlib import Path

import click
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from tqdm import tqdm

from unfccc_rag.utils import load_chunks

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)


class SingleQuestionAnswer(BaseModel):
    """A single question and answer pair grounded in a chunk."""

    answer: str
    question: str


SYSTEM_PROMPT = """You are a retrieval augmented generation benchmark expert.
Your are working with documents issued from the Conference Of Parties (COP) climate negociations as part of the UNFCCC.
Your goal is to generate a synthetic dataset of question/answer pairs based on document chunks which would allow us to evaluate the performance of our retrieval augmented generation system.
You have to generate exactly one concise and precise set of question and direct answer that can be answered to only using the provided chunk i.e. no external information should be required to answer the question.
Keep your questions relatively simple so that answers can be easily verified.
"""


def build_chunk_name(chunk: dict) -> str:
    """Create a descriptive name for a chunk for prompting."""
    meta = chunk.get("metadata", {})
    file_name = meta.get("file_name") or chunk.get("source_file") or "unknown_source"
    chunk_no = meta.get("chunk_number")
    if chunk_no is not None:
        return f"{file_name}#chunk-{chunk_no}"
    return str(file_name)


def build_prompt(document_name: str, document_text: str) -> str:
    """Build the prompt for the given chunk."""
    return f"Document name: {document_name}\n\nDocument contents:\n{document_text}"


def generate_question(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> SingleQuestionAnswer:
    """Generate one grounded Q/A for the given chunk."""
    agent_model = OpenAIChatModel(
        model_name=model,
        provider=OpenAIProvider(openai_client=client),  # type: ignore
    )

    agent = Agent(
        instructions=SYSTEM_PROMPT,
        model=agent_model,
        output_type=SingleQuestionAnswer,
        retries=5,
        output_retries=5,
    )

    response = agent.run_sync(
        user_prompt=prompt,
        model_settings=ModelSettings(
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.output  # type: ignore


@click.command()
@click.option(
    "--chunks-file",
    type=Path,
    required=True,
    help="Path to the chunks.json file.BufferError",
)
@click.option(
    "--output-file",
    type=Path,
    required=True,
    help="Output JSON file to write all generated Q/As pairs.",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="OpenAI model name to use for question generation.",
)
@click.option(
    "--api-key",
    type=str,
    required=True,
    help="OpenAI API key to use for question generation.",
)
@click.option(
    "--base-url",
    type=str,
    required=True,
    help="OpenAI base URL to use for question generation.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.5,
    help="Temperature for the OpenAI compatible model (default: 0.5).",
)
@click.option(
    "--max-tokens",
    type=int,
    default=8192,
    help="Maximum number of tokens to generate (default: 8192).",
)
def generate(
    chunks_file: Path,
    output_file: Path,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> None:
    """Generate a synthetic dataset of question/answer pairs based on document chunks."""

    chunks = load_chunks(chunks_file)
    if not chunks:
        logger.info(f"No chunks found in {chunks_file}")
        return

    logger.info(f"Found {len(chunks)} chunks in {chunks_file}.")

    # Optional progress bar
    try:
        progress_iter = tqdm(
            chunks.items(), total=len(chunks), desc="Generating Q/As", unit="chunk"
        )
    except Exception:
        progress_iter = chunks.items()

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=3,
    )

    # Stream results as a valid JSON array progressively
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        try:
            f.write("{\n")
            for chunk_id, chunk in progress_iter:
                name = build_chunk_name(chunk)
                text = chunk.get("text", "")
                prompt = build_prompt(name, text)
                qa = generate_question(
                    client,
                    model,
                    prompt,
                    temperature,
                    max_tokens,
                )
                out = dict(chunk)
                out["question"] = qa.question
                out["answer"] = qa.answer
                out["benchmark_qa_model"] = model
                if count > 0:
                    f.write(",\n")
                f.write(f'  "{chunk_id}":')
                pretty = json.dumps(out, ensure_ascii=False, indent=2)
                indented = pretty.replace("\n", "\n  ")
                f.write(indented)
                f.flush()
                count += 1
            f.write("\n}\n")
        except KeyboardInterrupt:
            logger.info("Interrupted, saving...")
            f.write("\n}\n")
    logger.info(f"Saved {count} Q/As to {output_file}")
