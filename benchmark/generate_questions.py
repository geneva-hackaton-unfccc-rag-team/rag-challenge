#!/usr/bin/env python3
"""Generate document-grounded questions via the OpenAI Responses API."""

import argparse
import json
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


class SingleQuestionAnswer(BaseModel):
    """A single question and answer pair grounded in a chunk."""

    answer: str
    question: str


SYSTEM_PROMPT = (
    "You are a retrieval augmented generation benchmark expert.\n"
    "Generate exactly one concise question and its direct answer that are SOLELY grounded\n"
    "in the provided chunk text. The question should be answerable using only the chunk.\n"
    'Respond ONLY with a JSON object of the form {"answer": string, "question": string}.'
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Read chunks.json and ask an OpenAI model for one grounded Q/A per chunk."
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI base URL to use for question generation.",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key to use for question generation.",
    )
    parser.add_argument(
        "--model",
        help="OpenAI model name to use for question generation.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "chunks.json",
        help="Path to the chunks.json file (defaults to ../data/chunks.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "chunks_questions.json",
        help="Output JSON file to write all generated Q/As.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for the OpenAI compatible model.",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from chunks.json."""
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
    max_tokens: int = 8192,
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

    response = agent.run_sync(user_prompt=prompt)
    return response.output  # type: ignore


def main() -> None:
    """Main function."""
    args = parse_args()

    chunks = load_chunks(args.chunks_path)
    if not chunks:
        print(f"No chunks found in {args.chunks_path}")
        return

    print(f"Found {len(chunks)} chunks in {args.chunks_path}.")

    # Optional progress bar
    try:
        from tqdm import tqdm  # type: ignore

        progress_iter = tqdm(
            chunks, total=len(chunks), desc="Generating Q/As", unit="chunk"
        )
    except Exception:
        progress_iter = chunks

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=3,
    )

    # Stream results as a valid JSON array progressively
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("[\n")
        for chunk in progress_iter:
            name = build_chunk_name(chunk)
            text = chunk.get("text", "")
            prompt = build_prompt(name, text)
            qa = generate_question(client, args.model, prompt)
            out = dict(chunk)
            out["question"] = qa.question
            out["answer"] = qa.answer
            if count > 0:
                f.write(",\n")
            pretty = json.dumps(out, ensure_ascii=False, indent=2)
            indented = "  " + pretty.replace("\n", "\n  ")
            f.write(indented)
            f.flush()
            count += 1
        f.write("\n]\n")
    print(f"Saved {count} Q/As to {args.output}")


if __name__ == "__main__":
    main()
