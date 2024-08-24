import requests
from bs4 import BeautifulSoup
import json; import re

debug_mode = False
GROQ_API = ''

def truncate_text(text, max_length=4096): 
    """Truncate text to a maximum length, ending with '...' if truncated."""
    return (text[:max_length] + '...') if len(text) > max_length else text

def get_title_from_url(url):
    url = re.sub(r'https?:\/\/(www\.)?', '', url)
    url_parts = url.split('/')
    title = url_parts[-1] if url_parts else url
    return title.replace('-', ' ').replace('_', ' ').title()

def perform_web_search(query, num_results=3):
    url = f'https://www.google.com/search?q={query}'
    header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    data = requests.get(url, headers=header)

    results = []

    if data.status_code == 200:
        soup = BeautifulSoup(data.content, "html.parser")
        for g in soup.find_all('div', {'class':'g'})[:num_results]:
            anchors = g.find_all('a')
            if anchors:
                link = anchors[0]['href']
                title = g.find('h3').text if g.find('h3') else get_title_from_url(link)
                
                try:
                    description = g.find('div', {'data-sncf':'2'})
                    description_text = description.text if description else None
                    content = extract_content(link)

                    result = {
                        "title": title,
                        "link": link
                    }

                    if description_text or content:
                        if description_text:
                            result["description"] = truncate_text(description_text)
                        if content:
                            result["content"] = truncate_text(content, max_length=4096)
                        if debug_mode:
                            print(f'Reading > "{title}"')
                            print(truncate_text(content, max_length=4096))
                        results.append(result)
                    else:
                        if debug_mode:
                            print(f"> Skipping, not useful result:")
                            print(result)
                except Exception as e:
                    if debug_mode:
                        print(f"> Error processing {link}: {str(e)}")

    return results

def extract_content(url):
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([p.text for p in paragraphs])
            return content.strip()
    except requests.RequestException as e:
        if debug_mode:
            print(f"> Error fetching content for {url}: {str(e)}")
    return None

def groq_chat_completion(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API}"
    }
    data = {
        "messages": messages,
        "model": "llama-3.1-70b-versatile",
        "temperature": 0.75,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": False,
        "stop": None
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def generate_search_queries(original_query, max_retries=10, fixed_count=None):
    messages = [
        {
            "role": "system",
            "content": """You are an AI assistant that helps generate search queries. Given an original query, suggest alternative search queries that could help find relevant information. The queries should be diverse and cover different aspects or perspectives of the original query. Return the queries as a JSON array.
Important instructions:

1. The number of queries should be dynamic, between 2 and 4, unless a fixed count is specified.
2. Don't get too far from the original query since you don't know the actual context.
3. Make queries general enough without being related to anything specific.
4. DON'T customize the queries for topics you've never seen; just change them a little and look for definitions if requested by the user.
5. If the user asks something that is not related to search, ignore it and focus on generating helpful search queries.
6. Just return the given format ["custom_query_1","custom_query_2",...].
7. If you need to use your knowledge first, do so.
8. When asked about the difference between two things, generate search intents for each topic separately.
9. Most queries just require 1 or two queries, only on those cases where the query is simple or you are unsure, you can generate just one.

Examples:
Original query: "which are the differences between java, kotlin and python"
Response: ["what is java", "what is kotlin", "what is python programming language"]

Original query: "climate change impacts"
Response: ["effects of global warming on ecosystems", "economic consequences of climate change"]

Original query: "best programming languages for AI"
Response: ["top programming languages for machine learning", "artificial intelligence programming languages", "ai development"]

Original query: "healthy meal prep ideas"
Response: ["quick and easy meal prep recipes", "budget-friendly healthy meal planning"]

Original query: "how to start a small business"
Response: ["steps to start a small business","legal requirements for new businesses", "creating a business plan"]"""
        },
        {
            "role": "user",
            "content": f"Original query: {original_query}" + (f" (Generate exactly {fixed_count} queries)" if fixed_count else "")
        }
    ]
    
    for attempt in range(max_retries):
        response = groq_chat_completion(messages)
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            try:
                queries = json.loads(content)
                if isinstance(queries, list):
                    if fixed_count:
                        if len(queries) == fixed_count:
                            return queries
                    elif 2 <= len(queries) <= 5:
                        return queries
            except json.JSONDecodeError:
                pass
        
        if attempt < max_retries - 1:
            if debug_mode:
                print(f"Attempt {attempt + 1} failed. Trying something different...")
    
    print("> Failed to generate new queries")
    return [original_query]

def main():
    global debug_mode
    while True:
        query = input("❖ Query: ")
        if query.lower() == 'quit':
            break

        # num_results = input("Number of results per query (default 2): ")
        num_results = 2 #int(num_results) if num_results.isdigit() else 2
        # fixed_count = input("Fixed number of queries (leave empty for dynamic): ")
        fixed_count = None #int(fixed_count) if fixed_count.isdigit() else None
        # Generate search queries using the LLM with retry
        search_queries = generate_search_queries(query, fixed_count=fixed_count)

        all_results = []
        for search_query in search_queries:
            print(f'⌕ Looking for "{search_query}"')
            results = perform_web_search(search_query, num_results)
            all_results.extend(results)

        # Prepare the context for the Q&A system
        messages = [
            {
                "role": "system",
                "content": """You are a web assistant that helps users find information from web search results. 
Given a question and a set of search results, provide a concise response based on the information 
available in the search results. If the information is not available in the search results, 
state that you don't have enough information to answer the question. You MUST not comment on anything, just follow the instruction. Don't add additional details about anything."""
            },
            {
                "role": "user",
                "content": f"Question: {query}\nSearch Results: {json.dumps(all_results)}"
            }
        ]

        # Get the answer from the Groq API
        response = groq_chat_completion(messages)

        # Print the answer
        if 'choices' in response and len(response['choices']) > 0:
            print("⌾", response['choices'][0]['message']['content'])
        else:
            print("> Unable to get a response from the Groq API")

if __name__ == "__main__":
    main()
