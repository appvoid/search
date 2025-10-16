# search

## Overview

This project provides a web search question answering system that can be used either from the command line or through an HTTP API. The assistant generates focused search queries, scrapes the public web for relevant passages, and synthesises an answer with Groq's LLaMA models. Math questions, simple trivia and realtime queries are routed through specialised flows to keep responses accurate and concise.

## Requirements

- Python 3.9+
- `pip install -r requirements.txt`
- A Groq API key with access to the LLaMA 3 models (`GROQ_API_KEY`).

## Configuration

Configuration is centralised in `search_core.config.Settings`. Values can be supplied via environment variables, a local `.env` file, or CLI flags when using the interactive client. The most relevant options are:

| Variable | Description | Default |
| --- | --- | --- |
| `GROQ_API_KEY` | Groq API key used for all LLM calls | **required** |
| `DEBUG_MODE` | Enables verbose logging in the CLI and server flows | `false` |
| `MAX_RETRIES` | Maximum number of search refinement loops | `3` |
| `SEARCH_RESULTS_PER_QUERY` | Number of web results fetched per generated query | `2` |
| `REQUEST_TIMEOUT` | Timeout (seconds) for outbound HTTP requests | `10.0` |
| `MAX_CONTENT_LENGTH` | Maximum length of extracted page content | `2048` |
| `QUERY_GENERATION_ATTEMPTS` | Attempts allowed when generating new search queries | `3` |
| `GROQ_MODEL` | Groq model used for completions | `llama-3.3-70b-versatile` |
| `GROQ_TEMPERATURE` | Sampling temperature for completions | `0.5` |
| `GROQ_MAX_TOKENS` | Maximum tokens generated per completion | `1024` |

All values can be overridden at runtime through `Settings.with_overrides(...)` and the CLI options documented below.

## Command Line Usage

The interactive assistant lives in `search.py` and now accepts configuration flags. The Groq API key can be provided either through the environment or directly via the CLI.

```bash
export GROQ_API_KEY="your-temporary-or-permanent-key"
python search.py --debug --max-retries 4
```

Common flags:

- `--api-key` – override the Groq API key for the current session
- `--debug` – print intermediate reasoning when answers are not satisfactory
- `--max-retries` – limit the number of refinement loops per question
- `--results-per-query` – tune how many webpages are retrieved per generated query
- `--timeout`, `--model`, `--temperature`, `--max-tokens` – fine-tune Groq request parameters

Exit the CLI with `quit`, `exit`, `Ctrl+D`, or `Ctrl+C`.

## HTTP API

`server.py` exposes a Flask application with an asynchronous `/ask` endpoint. Start the server with:

```bash
export GROQ_API_KEY="your-temporary-or-permanent-key"
python server.py
```

Send a request with:

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the current inflation rate in Canada?", "max_retries": 4}'
```

The response schema is:

```json
{
  "response": "...",  // assistant answer
  "type": "search"     // "search", "math", or "text"
}
```

If `max_retries` is omitted the server falls back to the configured default. Invalid input or missing API keys return descriptive error messages.

## Architecture

The heavy lifting lives inside the `search_core` package:

- `config.py` – centralised configuration loader (`Settings`)
- `groq.py` – resilient synchronous/asynchronous Groq API clients with shared error handling
- `searchers.py` – Google scraping logic for synchronous and asynchronous contexts
- `prompts.py` – curated prompt templates used throughout the pipeline
- `workflows.py` – orchestrators that coordinate query generation, evaluation, and answer synthesis
- `types.py` – lightweight dataclasses shared across modules

Both `search.py` and `server.py` compose these building blocks to provide consistent behaviour across the CLI and HTTP interfaces.

## Testing

Use the temporary key provided by Groq (or your own key) to perform manual smoke tests:

```bash
export GROQ_API_KEY="gsk_..."
python search.py --debug
```

For API testing, run the Flask server and issue a request with `curl` or a REST client.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
