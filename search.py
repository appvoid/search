import requests
from bs4 import BeautifulSoup
import json
import re

debug_mode = False
GROQ_API = ''

def truncate_text(text, max_length=1024): 
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
                            result["content"] = truncate_text(content, max_length=1024)
                        if debug_mode:
                            print(f'Reading > "{title}"')
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
        "max_tokens": 256,
        "top_p": 1,
        "stream": False,
        "stop": None
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def generate_search_queries(original_query):
    messages = [
        {
            "role": "system",
            "content": """You are an AI assistant that helps generate search queries. Given an original query, suggest 3 alternative search queries that could help find relevant information. The queries should be diverse and cover different aspects or perspectives of the original query. Return the queries as a JSON array but please DON'T get too far form the word since you don't know the actual context so make them general enough but without being related to anything specific, DON'T customize the queries for topics you never have seen just change it a little and look for definitions if requested by the user.

Examples:
Original query: "climate change impacts"
Response: ["effects of global warming on ecosystems", "economic consequences of climate change", "climate change mitigation strategies"]

Original query: "healthy breakfast ideas"
Response: ["quick and nutritious breakfast recipes", "benefits of eating a healthy breakfast", "low-carb breakfast options for weight loss"]

Original query: "artificial intelligence in healthcare"
Response: ["AI applications in medical diagnosis", "ethical concerns of AI in medicine", "machine learning for drug discovery"]"""
        },
        {
            "role": "user",
            "content": f"Original query: {original_query}"
        }
    ]
    
    response = groq_chat_completion(messages)
    
    if 'choices' in response and len(response['choices']) > 0:
        content = response['choices'][0]['message']['content']
        try:
            queries = json.loads(content)
            if isinstance(queries, list) and len(queries) == 3:
                return queries
        except json.JSONDecodeError:
            pass
    
    # Fallback to default queries if the LLM response is not as expected
    return [original_query, f"{original_query} explained", f"{original_query} examples"]

def main():
    global debug_mode
    while True:
        query = input("Search  > ")
        if query.lower() == 'quit':
            break

        num_results = '1'  # You can change this to input() if you want user input
        num_results = int(num_results) if num_results.isdigit() else 1 # one result makes it a lot faster

        # Generate search queries using the LLM
        search_queries = generate_search_queries(query)

        all_results = []
        for search_query in search_queries:
            print(f'Action  > Looking for "{search_query}"')
            results = perform_web_search(search_query, num_results)
            all_results.extend(results)

        # Prepare the context for the Q&A system
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant that helps users find information from web search results. 
Given a question and a set of search results, provide a concise answer based on the information 
available in the search results. If the information is not available in the search results, 
state that you don't have enough information to answer the question."""
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
            print("Answer  >", response['choices'][0]['message']['content'])
        else:
            print("> Unable to get a response from the Groq API")

if __name__ == "__main__":
    main()
