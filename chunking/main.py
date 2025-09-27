import json
from typing import List, Dict, Any
from pathlib import Path

MIN_CHUNK_SIZE = 20  # Minimum characters in a chunk
MAX_CHUNK_SIZE = 1000  # Maximum characters in a chunk


def chunk_document(file_path: Path) -> List[Dict[str, Any]]:
    chunks = []
    lines = file_path.read_text().splitlines()
    for line_num, line in enumerate(lines):
        cleaned_line = line.strip()

        # Skip empty lines and very short lines
        if len(cleaned_line) < MIN_CHUNK_SIZE:
            continue

        # Handle very long lines by splitting them
        if len(cleaned_line) > MAX_CHUNK_SIZE:
            raise ValueError(
                f"Line {line_num + 1} in {file_path} exceeds maximum length"
            )
        else:
            chunk = _create_chunk(cleaned_line, line_num, file_path)
            chunks.append(chunk)

    print(f"Processed {len(lines)} lines from {file_path}")
    print(f"Created {len(chunks)} chunks")
    return chunks


def _create_chunk(
    text: str, line_num: int, file_path: Path, chunk_num: int = None
) -> Dict[str, Any]:
    """Create a chunk dictionary with text and metadata."""
    chunk_id = f"line_{line_num}"
    if chunk_num:
        chunk_id += f"_part_{chunk_num}"

    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "line_number": line_num,
            "chunk_number": chunk_num if chunk_num else 1,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "chunk_type": "line",
            "char_count": len(text),
            "word_count": len(text.split()),
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

    chunks = chunk_document(input_file)

    output_dir = repo / "chunking/output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if chunks:
        output_file = output_dir / "CMA2016_1.1_Decisions_1_to_2_chunks.json"
        save_chunks(chunks, output_file)

        # Display some sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
            print(f"\nChunk {i + 1}:")
            print(f"  ID: {chunk['id']}")
            print(f"  Line: {chunk['metadata']['line_number']}")
            print(f"  Text: {chunk['text'][:100]}...")
            print(f"  Characters: {chunk['metadata']['char_count']}")

        print(f"\nTotal chunks created: {len(chunks)}")
    else:
        print("No chunks were created.")


if __name__ == "__main__":
    main()
