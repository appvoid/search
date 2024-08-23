import requests
from bs4 import BeautifulSoup
import json

JINA_API_KEY = ''
GROQ_API_KEY = ''

# The purpose of truncation is to not feed all the website content to the model
def truncate_text(text, max_length=512): 
    """Truncate text to a maximum length, ending with '...' if truncated."""
    return (text[:max_length] + '...') if len(text) > max_length else text

def extract_content(url):
    content_url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': f'Bearer {JINA_API_KEY}',
        'X-Return-Format': 'text'
    }
    try:
        response = requests.get(content_url, headers=headers, timeout=10)
        if response.status_code == 200 and response.text.strip():
            return response.text.strip()
    except requests.RequestException as e:
        print(f"Error fetching content for {url}: {str(e)}")
    return None

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
                title = g.find('h3').text if g.find('h3') else None
                
                try:
                    description = g.find('div', {'data-sncf':'2'})
                    description_text = description.text if description else None
                    content = extract_content(link)
                    
                    if description_text or content:
                        result = {
                            "title": title,
                            "link": link
                        }
                        if description_text:
                            result["description"] = truncate_text(description_text)
                        if content:
                            result["content"] = truncate_text(content, max_length=500)
                        results.append(result)
                    else:
                        print(f"Skipping result for {link}: No description or content")
                except Exception as e:
                    print(f"Error processing {link}: {str(e)}")

    return results

def groq_chat_completion(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    data = {
        "messages": messages,
        "model": "llama-3.1-70b-versatile",
        "temperature": 1,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": False,
        "stop": None
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def main():
    while True:
        query = input("Search > ")
        if query.lower() == 'quit':
            break

        num_results = input("Max # of articles > ")
        num_results = int(num_results) if num_results.isdigit() else 3

        # Perform web search
        search_results = perform_web_search(query, num_results)

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
                "content": f"Question: {query}\nSearch Results: {json.dumps(search_results)}"
            }
        ]

        # Get the answer from the Groq API
        response = groq_chat_completion(messages)

        # Print the answer
        print("\nQuery >", query)
        if 'choices' in response and len(response['choices']) > 0:
            print("Response > ", response['choices'][0]['message']['content'])
        else:
            print("Error: Unable to get a response from the Groq API.")
        print("\n" + "-"*64)

if __name__ == "__main__":
    main()