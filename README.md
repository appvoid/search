# search

### Overview
This project is a web search Q&A system that uses natural language processing (NLP) to answer user queries based on search results from the web. The system uses the Jina API to extract content from web pages and the Groq API to generate answers based on the search results.

### Features
- Web search using Google search results
- Answer generation using Groq API
- Support for multiple search results
- User-friendly interface for querying and displaying answers

### Requirements
- Python 3.8+
- `requests` library for making HTTP requests
- `BeautifulSoup` library for parsing HTML content
- Groq API key for answer generation

### Usage
1. Clone the repository and install the required libraries using pip install -r requirements.txt.
2. Replace the GROQ_API_KEY variable with your own API key.
3. Run the search.py file to start the Q&A system.
4. Enter a query when prompted.

After this, the system will display the answer based on the search results.

### License
This project is licensed under the MIT License. See the LICENSE file for more information.
