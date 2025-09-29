import logging
from pathlib import Path

import click
from tqdm import tqdm

from unfccc_rag.retrieval import load_retrieval_engine
from unfccc_rag.utils import load_chunks

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--chunks-file",
    type=Path,
    required=True,
    help="Path to the chunks file.",
)
@click.option(
    "--model-name",
    type=str,
    required=True,
    help="Name of the model to use for the embeddings.",
)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=5,
    multiple=True,
    help="Number of top chunks to retrieve.",
)
@click.option(
    "--alpha",
    "-a",
    type=float,
    default=0.9,
    multiple=True,
    help="Alpha for the retrieval engine.",
)
@click.option(
    "--output-file",
    "-o",
    type=Path,
    default=None,
    help="Output file to write the markdown table.",
)
def run(
    chunks_file: Path,
    model_name: str,
    top_k: tuple[int, ...],
    alpha: tuple[float, ...],
    output_file: Path | None,
):
    """Evaluate the retrieval engine."""
    if output_file and output_file.exists():
        raise click.ClickException(f"Output file {output_file} already exists!")

    markdown_table = (
        "| n-chunks | top-k | alpha | retrieval accuracy |\n|---|---|---|---|\n"
    )

    for _top_k in top_k:
        for _alpha in alpha:
            rag_query = load_retrieval_engine(
                embeddings_file=chunks_file,
                model_name=model_name,
                top_k=_top_k,
                alpha=_alpha,
                silent=True,
            )
            successful_retrievals: list[bool] = []

            chunks = load_chunks(chunks_file)
            pbar = tqdm(desc="Evaluating chunks", total=len(chunks))
            for chunk_id, chunk in chunks.items():
                question = chunk["question"]
                retrieved_chunks = rag_query(question)

                if chunk_id in retrieved_chunks:
                    successful_retrievals.append(True)
                else:
                    successful_retrievals.append(False)

                retrieval_acc = (
                    100 * sum(successful_retrievals) / len(successful_retrievals)
                )

                pbar.set_description(
                    f"retrieval_acc@k{_top_k}/a{_alpha}: {retrieval_acc:.1f}%"
                )
                pbar.update(1)
            markdown_table += (
                f"| {len(chunks)} | {_top_k} | {_alpha} | {retrieval_acc:.1f}% |\n"
            )
            pbar.close()

    if output_file:
        with open(output_file, "w") as f:
            f.write(markdown_table)
