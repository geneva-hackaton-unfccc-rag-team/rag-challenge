import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import GemmaTokenizerFast


MIN_TOKENS = 20
MAX_TOKENS = 1800

tokenizer = GemmaTokenizerFast.from_pretrained("hf-internal-testing/dummy-gemma")


def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def chunk_document(file_path: Path) -> List[Dict[str, Any]]:
    chunks = []
    lines = file_path.read_text().splitlines()
    line_start_i = 0
    cum_tokens = 0
    for line_end_i, line in enumerate(lines):
        n_tokens = num_tokens(line)
        # Create chunk with lines up to (but not including) current line
        if cum_tokens + n_tokens > MAX_TOKENS and cum_tokens > 0:
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

    print(f"Processed {len(lines)} lines from {file_path}")
    print(f"Created {len(chunks)} chunks")
    return chunks


def _create_chunk(
    text: str, line_start: int, line_end: int, chunk_i: int, file_path: Path
) -> Dict[str, Any]:
    return {
        "id": chunk_i,
        "text": text,
        "metadata": {
            "line_start": line_start,
            "line_end": line_end,
            "chunk_number": chunk_i,
            "file_path": str(file_path),
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
    print(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    # Define the input file path
    repo = Path(__file__).parent.parent

    input_file = repo / "data/CMA_txt/CMA2016_1.1_Decisions_1_to_2.txt"

    # Check if file exists
    if not input_file.exists():
        print(f"Error: File {input_file} not found!")
        return

    print(f"Processing document: {input_file}")
    chunks = chunk_document(input_file)

    output_dir = repo / "chunking/output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if chunks:
        output_file = output_dir / "CMA2016_1.1_Decisions_1_to_2_chunks.json"
        save_chunks(chunks, output_file)

        # Display some sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i + 1}:")
            print(f"  ID: {chunk['id']}")
            print(
                f"  Lines: {chunk['metadata']['line_start']}-{chunk['metadata']['line_end']}"
            )
            print(f"  Text: {chunk['text'][:100]}...")
            print(f"  Characters: {chunk['metadata']['char_count']}")
            print(f"  Tokens: {chunk['metadata']['token_count']}")

        print(f"\nTotal chunks created: {len(chunks)}")

        # Show token statistics
        total_tokens = sum(chunk["metadata"]["token_count"] for chunk in chunks)
        max_tokens = max(chunk["metadata"]["token_count"] for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        print(f"Token statistics:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Average tokens per chunk: {avg_tokens:.1f}")
        print(f"  Max tokens in a chunk: {max_tokens}")
        print(
            f"  Chunks at token limit: {sum(1 for chunk in chunks if chunk['metadata']['token_count'] >= MAX_TOKENS * 0.95)}"
        )
    else:
        print("No chunks were created.")


if __name__ == "__main__":
    main()
