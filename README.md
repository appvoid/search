# search

### Overview
This project is a web search Q&A system that uses natural language processing (NLP) to answer user queries based on search results from the web. The system uses LLaMA 3 70B to extract content from web pages and generate answers based on the search results and/or an instruction.

### Main Features

- Web Search Q&A
- Query Type Evaluation
- Math Query Handling
- Simple Query Handling
- Best Answer Selection

### Requirements
- Python 3.8+
- `requests` library for making HTTP requests
- `BeautifulSoup` library for parsing HTML content
- Groq API key for answer generation

### Server Usage Examples
1. Web Search Q&A (Default)
- Query: What is the capital of France?
  - The system will perform a web search and provide an answer based on the results.
2. Math Queries
- Query: What is 15 * 7 + 3?
  - The system will detect this as a math query and provide the calculated result.
3. Simple Queries
- Query: Who wrote "To Kill a Mockingbird"?
  - For well-known information, the system may provide a direct answer without web search.
4. Realtime Information
- Query: What's the current weather in New York?
  - The system will recognize this as requiring up-to-date information and perform a web search.
- Best Answer Selection
  - For complex queries, the system may generate multiple answers and select the best one.

### API Usage
Send a POST request to /ask endpoint with JSON payload:
```
{
  "query": "Your question here",
  "max_retries": 3  // Optional, default is 3
}
```
Response format:
```
{
  "response": "Answer to the query",
  "type": "search" // or "math" or "text"
}
```

Note: The system automatically determines the query type and processes accordingly.

### License
This project is licensed under the MIT License. See the LICENSE file for more information.
