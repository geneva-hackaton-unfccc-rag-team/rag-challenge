import json
import logging
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import click
from transformers import GemmaTokenizerFast

logger = logging.getLogger(__name__)

MIN_TOKENS = 20
MAX_TOKENS = 1800
REPO_ROOT = Path(__file__).parent.parent

tokenizer: GemmaTokenizerFast | None = None


@lru_cache()
def get_tokenizer() -> GemmaTokenizerFast:
    """Get the tokenizer."""
    return GemmaTokenizerFast.from_pretrained("hf-internal-testing/dummy-gemma")


def num_tokens(text: str) -> int:
    """Get the number of tokens in a text."""
    # Lazy load the tokenizer
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


DECISION_REGEX = re.compile(r"^\s*(Decision)|(Annex)")


def should_break_chunk(line: str, cum_token: int) -> bool:
    """Determine if a line should force a chunk break."""
    if cum_token > MAX_TOKENS:
        return True
    # Force break on new Decision
    if DECISION_REGEX.match(line):
        return True
    return False


def chunk_document(file_path: Path) -> List[Dict[str, Any]]:
    chunks = []
    lines = file_path.read_text().splitlines()
    line_start_i = 0
    cum_tokens = 0
    for line_end_i, line in enumerate(lines):
        n_tokens = num_tokens(line)
        break_chunk = should_break_chunk(line, cum_tokens + n_tokens)
        # Create chunk with lines up to (but not including) current line
        if break_chunk:
            chunk_text = "\n".join(lines[line_start_i:line_end_i])
            chunk = _create_chunk(
                chunk_text, line_start_i, line_end_i - 1, len(chunks), file_path
            )
            chunks.append(chunk)
            line_start_i = line_end_i
            cum_tokens = n_tokens
        else:
            cum_tokens += n_tokens

        # Handle edge case: single line exceeds MAX_TOKENS
        if n_tokens > MAX_TOKENS:
            # Force create a chunk with just this line
            chunk_text = line
            chunk = _create_chunk(
                chunk_text, line_end_i, line_end_i, len(chunks), file_path
            )
            chunks.append(chunk)
            line_start_i = line_end_i + 1
            cum_tokens = 0

    if line_end_i != len(lines) - 1:
        raise ValueError("Logic error: not all lines processed")

    logger.info(f"Processed {len(lines)} lines from {file_path}")
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def _create_chunk(
    text: str, line_start: int, line_end: int, chunk_i: int, file_path: Path
) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "metadata": {
            "line_start": line_start,
            "line_end": line_end,
            "chunk_number": chunk_i,
            "file_path": str(file_path.relative_to(REPO_ROOT)),
            "file_name": file_path.name,
            "chunk_type": "line",
            "char_count": len(text),
            "word_count": len(text.split()),
            "token_count": num_tokens(text),
        },
    }


def save_chunks(chunks: List[Dict[str, Any]], output_path: str):
    """Save chunks to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


@click.command()
@click.option(
    "--data-dir",
    type=Path,
    required=True,
    help="Path to the data directory.",
)
@click.option(
    "--output-dir",
    type=Path,
    required=True,
    help="Path to the output directory.",
)
def chunk_documents(data_dir: Path, output_dir: Path):
    """Chunk txt documents in the data directory."""
    # Find all text files matching data/*/*.txt pattern
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} not found!")
        raise click.ClickException(f"Data directory {data_dir} not found!")

    # Find all .txt files in subdirectories of data/
    txt_files = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            txt_files.extend(subdir.glob("*.txt"))

    if not txt_files:
        logger.error("No .txt files found in data/*/ directories!")
        raise click.ClickException("No .txt files found in data/*/ directories!")

    logger.info(f"Found {len(txt_files)} text files to process:")
    for file in txt_files:
        logger.info(f"  - {file.relative_to(REPO_ROOT)}")

    all_chunks = []
    total_files_processed = 0

    for input_file in txt_files:
        logger.info("\n" + "=" * 60)
        logger.info(f"Processing document: {input_file.relative_to(REPO_ROOT)}")
        logger.info("=" * 60)

        chunks = chunk_document(input_file)

        if not chunks:
            logger.info(f"No chunks created for {input_file}, skipping.")
            continue

        # Add file source info to each chunk
        for chunk in chunks:
            chunk["source_file"] = str(input_file.relative_to(REPO_ROOT))

        all_chunks.extend(chunks)
        total_files_processed += 1

        # Display some sample chunks for this file
        logger.info(f"\nSample chunks from {input_file.name}:")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            logger.info(f"\nChunk {i + 1}:")
            logger.info(f"  ID: {chunk['id']}")
            logger.info(
                f"  Lines: {chunk['metadata']['line_start']}-{chunk['metadata']['line_end']}"
            )
            logger.info(f"  Text: {chunk['text'][:100]}...")
            logger.info(f"  Characters: {chunk['metadata']['char_count']}")
            logger.info(f"  Tokens: {chunk['metadata']['token_count']}")

        logger.info(f"Created {len(chunks)} chunks for this file")

    # Save combined chunks from all files
    if not all_chunks:
        logger.info("\nNo chunks were created from any files.")
        exit(0)

    combined_output_file = output_dir / "chunks.json"
    save_chunks(all_chunks, combined_output_file)

    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Files processed: {total_files_processed}/{len(txt_files)}")
    logger.info(f"Total chunks created: {len(all_chunks)}")

    # Show overall token statistics
    total_tokens = sum(chunk["metadata"]["token_count"] for chunk in all_chunks)
    max_tokens = max(chunk["metadata"]["token_count"] for chunk in all_chunks)
    avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0
    logger.info("\nOverall token statistics:")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Average tokens per chunk: {avg_tokens:.1f}")
    logger.info(f"  Max tokens in a chunk: {max_tokens}")
    logger.info(
        f"  Chunks at token limit: {sum(1 for chunk in all_chunks if chunk['metadata']['token_count'] >= MAX_TOKENS * 0.95)}"
    )

    # Show chunks per file
    logger.info("\nChunks per file:")
    file_chunk_counts = {}
    for chunk in all_chunks:
        source = chunk["source_file"]
        file_chunk_counts[source] = file_chunk_counts.get(source, 0) + 1

    for source, count in sorted(file_chunk_counts.items()):
        logger.info(f"  {source}: {count} chunks")

    # Generate a debug file
    output_file = REPO_ROOT / "data" / "debug.txt"
    with output_file.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk["text"] + "\n" + "=" * 100 + "\n")
