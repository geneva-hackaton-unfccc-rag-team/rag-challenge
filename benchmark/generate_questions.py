#!/usr/bin/env python3
"""Generate document-grounded questions via the OpenAI Responses API."""

import argparse
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


class QuestionAnswerPair(BaseModel):
    """A question and answer pair for a retrieval augmented generation benchmark."""

    answer: str
    question: str
    id: int


class BenchmarkQuestions(BaseModel):
    """A list of question and answer pairs for a retrieval augmented generation benchmark."""

    question_answer_pairs: list[QuestionAnswerPair]


N_QUESTIONS = 10
SYSTEM_PROMPT = f"""
Your are a retrieval augmented generation benchmark expert.
Your role is to generate a list of questions that are SOLELY grounded in the provided document.
Meaning if someone reads the document, they should be able to answer the questions without any other information.
You should generate a set of {N_QUESTIONS} questions that are concise and to the point.
Write your response in JSON format and only include the JSON object, no other text so that it can be parsed easily.
"""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="List .txt files in data/ and ask an OpenAI model for grounded questions."
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
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory that contains the .txt corpus (defaults to ../data relative to this script).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for the OpenAI compatible model.",
    )
    return parser.parse_args()


def list_text_files(root: Path) -> list[Path]:
    """List all .txt files in the given root directory.

    Args:
        root (Path): The root directory to list .txt files from.

    Returns:
        list[Path]: A list of paths to the .txt files.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def load_content(path: Path) -> str:
    """Load the content of the given path.

    Args:
        path (Path): The path to load the content from.

    Returns:
        str: The content of the given path.
    """
    return path.read_text(encoding="utf-8")


def build_prompt(document_name: str, document_text: str) -> str:
    """Build the prompt for the given document.

    Args:
        document_name (str): The name of the document.
        document_text (str): The content of the document.

    Returns:
        str: The prompt for the given document.
    """
    return f"""Document name: {document_name}\n"
Document contents:\n + {document_text}"""


def generate_questions(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 8192,
) -> BenchmarkQuestions:
    """Generate questions for the given document.

    Args:
        client (AsyncOpenAI): The OpenAI client to use.
        model (str): The model to use.
        prompt (str): The prompt to use.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        BenchmarkQuestions: The generated questions.
    """
    agent_model = OpenAIChatModel(
        model_name=model,
        provider=OpenAIProvider(openai_client=client),  # type: ignore
    )

    agent = Agent(
        instructions=SYSTEM_PROMPT,
        model=agent_model,
        output_type=BenchmarkQuestions,
        retries=5,
        output_retries=5,
    )

    response = agent.run_sync(user_prompt=prompt)
    return response.output  # type: ignore


def main() -> None:
    """Main function."""
    args = parse_args()

    files = list_text_files(args.data_root)
    if not files:
        print(f"No .txt files found under {args.data_root}")
        return

    print("Found the following .txt files:")
    for path in files:
        print(f" - {path.relative_to(args.data_root)}")

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=3,
    )

    for path in files:
        content = load_content(path)
        prompt = build_prompt(path.name, content)
        print("\n== Questions for", path.relative_to(args.data_root), "==")
        questions = generate_questions(client, args.model, prompt)
        questions_filename = path.with_name(f"{path.stem}_questions.json")
        with open(questions_filename, "w", encoding="utf-8") as f:
            f.write(questions.model_dump_json(indent=2))
        print(f"Saved questions to {questions_filename}")


if __name__ == "__main__":
    main()
