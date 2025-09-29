import logging
import re
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Any

import click
from transformers import GemmaTokenizerFast

from unfccc_rag.utils import get_chunk_id, save_chunks

logger = logging.getLogger(__name__)

MIN_TOKENS = 20
MAX_TOKENS = 1800

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


def chunk_document(file_path: Path) -> dict[str, Any]:
    chunks = {}
    lines = file_path.read_text().splitlines()
    line_start_i = 0
    cum_tokens = 0
    chunk_num = 0
    for line_end_i, line in enumerate(lines):
        n_tokens = num_tokens(line)
        break_chunk = should_break_chunk(line, cum_tokens + n_tokens)
        # Create chunk with lines up to (but not including) current line
        if break_chunk:
            chunk_text = "\n".join(lines[line_start_i:line_end_i])
            chunk_num += 1
            chunk = _create_chunk(
                chunk_text,
                line_start_i,
                line_end_i - 1,
                len(chunks),
                file_path,
                chunk_num,
            )
            chunks[get_chunk_id()] = chunk
            line_start_i = line_end_i
            cum_tokens = n_tokens
        else:
            cum_tokens += n_tokens

        # Handle edge case: single line exceeds MAX_TOKENS
        if n_tokens > MAX_TOKENS:
            # Force create a chunk with just this line
            chunk_text = line
            chunk_num += 1
            chunk = _create_chunk(
                chunk_text,
                line_end_i,
                line_end_i,
                len(chunks),
                file_path,
                chunk_num,
            )
            chunks[get_chunk_id()] = chunk
            line_start_i = line_end_i + 1
            cum_tokens = 0

    if line_end_i != len(lines) - 1:
        raise ValueError("Logic error: not all lines processed")

    logger.info(f"Processed {len(lines)} lines from {file_path}")
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def _create_chunk(
    text: str,
    line_start: int,
    line_end: int,
    chunk_i: int,
    file_path: Path,
    chunk_num: int,
) -> dict[str, Any]:
    return {
        "text": text,
        "line_start": line_start,
        "line_end": line_end,
        "chunk_number": chunk_i,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "chunk_type": "line",
        "char_count": len(text),
        "word_count": len(text.split()),
        "token_count": num_tokens(text),
        "chunk_number": chunk_num,
    }


@click.command()
@click.option(
    "--data-dir",
    type=Path,
    required=True,
    help="Path to the data directory.",
)
@click.option(
    "--output-file",
    type=Path,
    required=True,
    help="Path to the output file.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Whether to save a debug file.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Whether to overwrite the output file if it already exists.",
)
def chunk_documents(data_dir: Path, output_file: Path, debug: bool, overwrite: bool):
    """Chunk txt documents in the data directory."""
    # Find all text files matching data/*/*.txt pattern
    if not data_dir.exists():
        raise click.ClickException(f"Data directory {data_dir} not found!")

    if output_file.exists() and not overwrite:
        raise click.ClickException(f"Output file {output_file} already exists!")

    # Find all .txt files in subdirectories of data/
    txt_files = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            txt_files.extend(subdir.glob("*.txt"))

    if not txt_files:
        raise click.ClickException("No .txt files found in data/*/ directories!")

    logger.info(f"Found {len(txt_files)} text files to process:")
    for file in txt_files:
        logger.info(f"  - {file}")

    all_chunks = {}
    total_files_processed = 0

    for input_file in txt_files:
        logger.info("\n" + "=" * 60)
        logger.info(f"Processing document: {input_file}")
        logger.info("=" * 60)

        chunks = chunk_document(input_file)

        if not chunks:
            logger.info(f"No chunks created for {input_file}, skipping.")
            continue

        # Add file source info to each chunk
        for chunk_id in chunks:
            chunks[chunk_id]["source_file"] = str(input_file)

        all_chunks.update(chunks)
        total_files_processed += 1

        # Display some sample chunks for this file
        logger.info(f"\nSample chunks from {input_file.name}:")
        for i, (chunk_id, chunk) in enumerate(
            islice(chunks.items(), 2)
        ):  # Show first 2 chunks
            logger.info(f"\nChunk {i + 1}:")
            logger.info(f"  ID: {chunk_id}")
            logger.info(f"  Lines: {chunk['line_start']}-{chunk['line_end']}")
            logger.info(f"  Text: {chunk['text'][:100]}...")
            logger.info(f"  Characters: {chunk['char_count']}")
            logger.info(f"  Tokens: {chunk['token_count']}")

        logger.info(f"Created {len(chunks)} chunks for this file")

    # Save combined chunks from all files
    if not all_chunks:
        logger.info("\nNo chunks were created from any files.")
        exit(0)

    save_chunks(all_chunks, output_file)

    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Files processed: {total_files_processed}/{len(txt_files)}")
    logger.info(f"Total chunks created: {len(all_chunks)}")

    # Show overall token statistics
    total_tokens = sum(chunk["token_count"] for chunk in all_chunks.values())
    max_tokens = max(chunk["token_count"] for chunk in all_chunks.values())
    avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0
    logger.info("\nOverall token statistics:")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Average tokens per chunk: {avg_tokens:.1f}")
    logger.info(f"  Max tokens in a chunk: {max_tokens}")
    logger.info(
        f"  Chunks at token limit: {sum(1 for chunk in all_chunks.values() if chunk['token_count'] >= MAX_TOKENS * 0.95)}"
    )

    # Show chunks per file
    logger.info("\nChunks per file:")
    file_chunk_counts = {}
    for chunk in all_chunks.values():
        source = chunk["source_file"]
        file_chunk_counts[source] = file_chunk_counts.get(source, 0) + 1

    for source, count in sorted(file_chunk_counts.items()):
        logger.info(f"  {source}: {count} chunks")

    # Generate a debug file
    if debug:
        debug_file = output_file.with_suffix(".debug.txt")
        with debug_file.open("w", encoding="utf-8") as f:
            for chunk in all_chunks.values():
                f.write(chunk["text"] + "\n" + "=" * 100 + "\n")
