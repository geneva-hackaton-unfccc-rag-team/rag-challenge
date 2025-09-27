import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import GemmaTokenizerFast


MIN_TOKENS = 20
MAX_TOKENS = 1800
REPO_ROOT = Path(__file__).parent.parent

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
    print(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    # Find all text files matching data/*/*.txt pattern
    data_dir = REPO_ROOT / "data"
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} not found!")
        return
    
    # Find all .txt files in subdirectories of data/
    txt_files = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            txt_files.extend(subdir.glob("*.txt"))
    
    if not txt_files:
        print("No .txt files found in data/*/ directories!")
        return
    
    print(f"Found {len(txt_files)} text files to process:")
    for file in txt_files:
        print(f"  - {file.relative_to(REPO_ROOT)}")
    
    output_dir = REPO_ROOT / "data"
    
    all_chunks = []
    total_files_processed = 0
    
    for input_file in txt_files:
        print('\n' + '='*60)
        print(f"Processing document: {input_file.relative_to(REPO_ROOT)}")
        print('='*60)
        
        chunks = chunk_document(input_file)
        
        if not chunks:
            print(f"No chunks created for {input_file}, skipping.")
            continue

        # Add file source info to each chunk
        for chunk in chunks:
            chunk['source_file'] = str(input_file.relative_to(REPO_ROOT))
        
        all_chunks.extend(chunks)
        total_files_processed += 1
        
        # Display some sample chunks for this file
        print(f"\nSample chunks from {input_file.name}:")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"\nChunk {i + 1}:")
            print(f"  ID: {chunk['id']}")
            print(f"  Lines: {chunk['metadata']['line_start']}-{chunk['metadata']['line_end']}")
            print(f"  Text: {chunk['text'][:100]}...")
            print(f"  Characters: {chunk['metadata']['char_count']}")
            print(f"  Tokens: {chunk['metadata']['token_count']}")

        print(f"Created {len(chunks)} chunks for this file")
    
    # Save combined chunks from all files
    if not all_chunks:
        print("\nNo chunks were created from any files.")
        exit(0)

    combined_output_file = output_dir / "chunks.json"
    save_chunks(all_chunks, combined_output_file)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {total_files_processed}/{len(txt_files)}")
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Show overall token statistics
    total_tokens = sum(chunk["metadata"]["token_count"] for chunk in all_chunks)
    max_tokens = max(chunk["metadata"]["token_count"] for chunk in all_chunks)
    avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0
    print("\nOverall token statistics:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Average tokens per chunk: {avg_tokens:.1f}")
    print(f"  Max tokens in a chunk: {max_tokens}")
    print(f"  Chunks at token limit: {sum(1 for chunk in all_chunks if chunk['metadata']['token_count'] >= MAX_TOKENS * 0.95)}")
    
    # Show chunks per file
    print("\nChunks per file:")
    file_chunk_counts = {}
    for chunk in all_chunks:
        source = chunk['source_file']
        file_chunk_counts[source] = file_chunk_counts.get(source, 0) + 1
    
    for source, count in sorted(file_chunk_counts.items()):
        print(f"  {source}: {count} chunks")


if __name__ == "__main__":
    main()
