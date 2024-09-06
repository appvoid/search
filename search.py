import requests
from bs4 import BeautifulSoup
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
GROQ_API = ''
debug_mode = False

# Utility Functions
def truncate_text(text, max_length=2048):
    return (text[:max_length] + '...') if len(text) > max_length else text

def get_title_from_url(url):
    url = re.sub(r'https?:\/\/(www\.)?', '', url)
    url_parts = url.split('/')
    title = url_parts[-1] if url_parts else url
    return title.replace('-', ' ').replace('_', ' ').title()

# Abstracted Functionality Modules
class WebSearchModule:
    def __init__(self):
        self.session = requests.Session()
        self.header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}

    def perform_search(self, query, num_results=2):
        url = f'https://www.google.com/search?q={query}'
        data = self.session.get(url, headers=self.header)

        results = []

        if data.status_code == 200:
            soup = BeautifulSoup(data.content, "html.parser")
            with ThreadPoolExecutor(max_workers=num_results) as executor:
                futures = []
                for g in soup.find_all('div', {'class':'g'})[:num_results]:
                    anchors = g.find_all('a')
                    if anchors:
                        link = anchors[0]['href']
                        title = g.find('h3').text if g.find('h3') else get_title_from_url(link)
                        description = g.find('div', {'data-sncf':'2'})
                        description_text = description.text if description else None
                        futures.append(executor.submit(self.process_result, title, link, description_text))
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)

        return results

    def process_result(self, title, link, description_text):
        try:
            content = self.extract_content(link)
            result = {
                "title": title,
                "link": link
            }
            if description_text or content:
                if description_text:
                    result["description"] = truncate_text(description_text)
                if content:
                    result["content"] = truncate_text(content, max_length=2048)
                if debug_mode:
                    print(f'Reading > "{title}"')
                    print(truncate_text(content, max_length=2048))
                return result
            elif debug_mode:
                print(f"> Skipping, not useful result:")
                print(result)
        except Exception as e:
            if debug_mode:
                print(f"> Error processing {link}: {str(e)}")
        return None

    def extract_content(self, url):
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                content = ' '.join([p.text for p in paragraphs])
                return content.strip()
        except requests.RequestException as e:
            if debug_mode:
                print(f"> Error fetching content for {url}: {str(e)}")
        return None

class GroqAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        })

    def chat_completion(self, messages):
        url = "https://api.groq.com/openai/v1/chat/completions"
        data = {
            "messages": messages,
            "model": "llama-3.1-70b-versatile",
            "temperature": 0.75,
            "max_tokens": 1024,
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        response = self.session.post(url, json=data)
        return response.json()

# Main Logic
def generate_search_queries(groq_api, original_query, max_retries=3, fixed_count=None, previous_queries=None, previous_answer=None):
    system_content = """You are an AI assistant that helps generate search queries. Given an original query, suggest alternative search queries that could help find relevant information. The queries should be diverse and cover different aspects or perspectives of the original query. Return the queries as a JSON array.
    Important instructions:
    
    1. The number of queries should be dynamic, between 2 and 4, unless a fixed count is specified.
    2. Don't get too far from the original query since you don't know the actual context.
    3. Make queries general enough without being related to anything specific.
    4. DON'T customize the queries for topics you've never seen; just change them a little and look for definitions if requested by the user.
    5. If the user asks something that is not related to search, ignore it and focus on generating helpful search queries.
    6. Just return the given format ["custom_query_1","custom_query_2",...].
    7. If you need to use your knowledge first, do so.
    8. When asked about the difference between two things, generate search intents for each topic separately.
    9. ALWAYS at most queries just require 1 or two queries, only on those cases where the query is simple or you are unsure, generate more than one or two.
    10. If previous queries and an answer are provided, generate new queries that address the shortcomings of the previous answer and avoid repeating the previous queries."""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Original query: {original_query}" + (f" (Generate exactly {fixed_count} queries)" if fixed_count else "")}
    ]

    if previous_queries and previous_answer:
        messages.append({
            "role": "user",
            "content": f"Previous queries: {previous_queries}\nPrevious answer: {previous_answer}\nPlease generate new queries to address any shortcomings in the previous answer."
        })

    for attempt in range(max_retries):
        response = groq_api.chat_completion(messages)

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

        if attempt < max_retries - 1 and debug_mode:
            print(f"Attempt {attempt + 1} failed. Trying something different...")

    print("> Failed to generate new queries")
    return [original_query]

def evaluate_answer(groq_api, query, answer):
    messages = [
        {"role": "system", "content": """You are an AI assistant that evaluates the quality and completeness of its own answer to user queries. 
    Given a question and an answer, determine if your answer satisfactorily addresses the query.
    Respond with a JSON object containing two fields:
    1. "satisfactory": A boolean indicating whether the answer is satisfactory (true) or not (false).
    2. "reason": A brief explanation of why your thought is or is not satisfactory. Like "I will keep looking for information since last thought is not addressing the query because..." or "Let look for something different. My last search didn't solve the query. The reason is..." or "I found the answer! The reason is..." or just provide the reason but be creative with the words choosen for the reason. You can be flexible, meaning that you can set as satisfactory if it is satisfactory enough and completes one of the core reasons for the search itself.
    Only return the JSON object, no additional text."""},
        {"role": "user", "content": f"Question: {query}\nAnswer: {answer}"}
    ]

    response = groq_api.chat_completion(messages)

    if 'choices' in response and len(response['choices']) > 0:
        content = response['choices'][0]['message']['content']
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("> Error parsing evaluation response")
    
    return {"satisfactory": False, "reason": "Unable to evaluate the answer"}

def main():
    global debug_mode
    search_module = WebSearchModule()
    groq_api = GroqAPI(GROQ_API)

    while True:
        query = input("❖ Query: ")
        if query.lower() == 'quit':
            break

        num_results = 1
        fixed_count = None
        previous_queries = None
        previous_answer = None

        while True:
            search_queries = generate_search_queries(groq_api, query, fixed_count=fixed_count, previous_queries=previous_queries, previous_answer=previous_answer)

            with ThreadPoolExecutor(max_workers=len(search_queries)) as executor:
                futures = [executor.submit(search_module.perform_search, sq, num_results) for sq in search_queries]
                all_results = []
                for future, sq in zip(as_completed(futures), search_queries):
                    print(f'⌕ Looking for "{sq}"')
                    all_results.extend(future.result())

            messages = [
                {"role": "system", "content": """You are a web assistant that helps users find information from web search results. 
    Given a question and a set of search results, provide a concise response based on the information 
    available in the search results. If the information is not available in the search results, 
    state that you don't have enough information to answer the question. You MUST not comment on anything, just follow the instruction. Don't add additional details about anything."""},
                {"role": "user", "content": f"Question: {query}\nSearch Results: {json.dumps(all_results)}"}
            ]

            response = groq_api.chat_completion(messages)

            if 'choices' in response and len(response['choices']) > 0:
                answer = response['choices'][0]['message']['content']
                evaluation = evaluate_answer(groq_api, query, answer)
                if evaluation["satisfactory"]:
                    print("⌾", answer)
                    break
                else:
                    print(f"✗ {evaluation['reason']}")
                    previous_queries = search_queries
                    previous_answer = answer
            else:
                print("> Unable to get a response from the Groq API")
                break

if __name__ == "__main__":
    main()
