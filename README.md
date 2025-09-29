# rag-challenge


Welcome to the UNFCCC RAG System!

## Setup

You'll need to have `uv` installed.
Check out the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for more information.

Most unix systems can install it using:
```bash
curl -fsSL https://astral.sh/uv/install.sh | sh
```

Then, you can install the project dependencies using:
```bash
uv sync
```

## Dataset

The UNFCCC dataset can be found in the `data` directory.
It is a collection of txt files that contain the text of the UNFCCC documents.
For now, it has been provided as is; but we could eventually try to write a clever scraper to get the documents.

## Chunking Documents

The first step to build the RAG system is to chunk the documents.

```shell
uv run chunk-txt-documents --data-dir data/ --output-file data/chunks.json
```

## Computing embeddings

The next step is to compute the embeddings of the chunks.

```shell
uv run serialize-embeddings --chunks-file data/chunks.json --output-file data/chunks.json --overwrite
```

## Benchmarking

To benchmark the system, we can create a synthetic dataset of question/answer pairs based on the chunk's content.
The dataset can be generated using the following command:

```shell
uv run generate-synthetic-dataset --chunks-file data/chunks.json --output-file data/chunks.json --model openai/gpt-oss-20b --api-key x --base-url http://127.0.0.1:1234
```

An LLM served using an OpenAI compatible API is used to generate the synthetic dataset.
The question/answer pairs dataset has been generated using the OpenAI gpt-oss-20b model served locally using [LMStudio](https://lmstudio.ai/).


## Web App

A simple chatbot interface built with Gradio that demonstrates a streaming response pattern.

```bash
uv run start-web-app
```

Open your browser and navigate to the URL shown in the terminal (typically `http://127.0.0.1:7860`)

## MCP Server

A simple MCP server that can be used to retrieve documents from the UNFCCC dataset.

The MCP configuration for the client should look like:

```json
{
  "mcpServers": {
    "unfccc-rag-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/path/to/rag-challenge",
        "start-rag-server",
        "--embeddings-file",
        "/path/to/rag-challenge/data/chunks.json"
      ],
      "env": {
        "HF_TOKEN": "<YOUR_HF_TOKEN>"
      }
    }
  }
}
```

A development server can be started using:

```shell
uv run mcp dev unfccc-rag/mcp.py
```