# server.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import aiohttp
from bs4 import BeautifulSoup
import json
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
GROQ_API_KEY = ''
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))  # Default to 3 if not set

# Utility Functions
def truncate_text(text, max_length=2500):
    return (text[:max_length] + '...') if len(text) > max_length else text

def get_title_from_url(url):
    url = re.sub(r'https?:\/\/(www\.)?', '', url)
    url_parts = url.split('/')
    title = url_parts[-1] if url_parts else url
    return title.replace('-', ' ').replace('_', ' ').title()

# Query Type Evaluator
class QueryTypeEvaluator:
    def __init__(self, groq_api):
        self.groq_api = groq_api

    async def evaluate_query_type(self, session, query):
        messages = [
            {"role": "system", "content": """You are an Web assistant that evaluates the type of query a user asks. 
            Categorize the query into one of the following types:
            1. simple: if it can be answered with general knowledge or information that is typically well-known on the internet, please provide a short answer as relevant as possible from the llm itself, but make sure you are completly sure you know the answer, don't make things up.
            2. realtime: if it requires up-to-date information like the current date, time, or recent events, or the user explicitly asks you to look on the internet you should state as: realtime
            3. math: if it involves ANY kind of mathematical calculations. Every math question be it counting letters or complex formulas.

            Remember to prioritize realtime over anything else if you are not sure about something. Realtime is like your default.
             
            Respond with the category as a single word ("simple", "realtime", or "math") without any additional text."""},
            {"role": "user", "content": f"Query: {query}"}
        ]

        response = await self.groq_api.chat_completion(session, messages)

        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content'].strip().lower()
        return "simple"  # Default to "simple" if unable to evaluate

# Web Search Module
class WebSearchModule:
    def __init__(self):
        self.header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}

    async def perform_search(self, session, query, num_results=3):
        url = f'https://www.google.com/search?q={query}'
        async with session.get(url, headers=self.header) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                results = []
                tasks = []
                for g in soup.find_all('div', {'class': 'g'})[:num_results]:
                    anchors = g.find_all('a')
                    if anchors:
                        link = anchors[0]['href']
                        title = g.find('h3').text if g.find('h3') else get_title_from_url(link)
                        description = g.find('div', {'data-sncf': '2'})
                        description_text = description.text if description else None
                        tasks.append(self.process_result(session, title, link, description_text))
                results = await asyncio.gather(*tasks)
                return [r for r in results if r]
            return []

    async def process_result(self, session, title, link, description_text):
        try:
            content = await self.extract_content(session, link)
            result = {
                "title": title,
                "link": link
            }
            if description_text or content:
                if description_text:
                    result["description"] = truncate_text(description_text)
                if content:
                    result["content"] = truncate_text(content, max_length=2500)
                if DEBUG_MODE:
                    print(f'Reading > "{title}"')
                    print(truncate_text(content, max_length=2500))
                return result
            elif DEBUG_MODE:
                print(f"> Skipping, not useful result:")
                print(result)
        except Exception as e:
            if DEBUG_MODE:
                print(f"> Error processing {link}: {str(e)}")
        return None

    async def extract_content(self, session, url):
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    paragraphs = soup.find_all('p')
                    content = ' '.join([p.text for p in paragraphs])
                    return content.strip()
        except Exception as e:
            if DEBUG_MODE:
                print(f"> Error fetching content for {url}: {str(e)}")
        return None

# Groq API Module
class GroqAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def chat_completion(self, session, messages):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "messages": messages,
            # "model": "llama-3.2-90b-text-preview",
            "model": "llama-3.1-70b-versatile",
            # "model": "llama-3.2-11b-text-preview",
            "temperature": 0.5,
            "max_tokens": 1024,
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        async with session.post(self.base_url, headers=headers, json=data) as response:
            return await response.json()

# Main Logic Functions
async def generate_search_queries(groq_api, session, original_query, max_retries=3, fixed_count=None, previous_queries=None, previous_answer=None):
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
    9. ALWAYS at most queries just require one or two queries, only on those cases where the query is simple or you are unsure, generate more than one or two.
    10. If previous queries and an answer are provided, generate new queries that address the shortcomings of the previous answer and avoid repeating the previous queries.
    11. ALWAYS split searches for each important part of the query in case you need to gather information but make sure to not get off the rails. In short, don't look for things together, make a search for each important part instead. DONT LOOK FOR THINGS TOGETHER."""

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
        response = await groq_api.chat_completion(session, messages)

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

        if attempt < max_retries - 1 and DEBUG_MODE:
            print(f"Attempt {attempt + 1} failed. Trying something different...")

    print("> Failed to generate new queries")
    return [original_query]

async def evaluate_answer(groq_api, session, query, answer):
    messages = [
        {"role": "system", "content": """You are an AI assistant that evaluates the quality and completeness of its own answer to user queries. 
    Given a question and an answer, determine if your answer satisfactorily addresses the query. You are highly tolerant to answers that are close to the intent so if it is close enough, you can say is satisfactory. Remember, if it's close enough, mark it as satisfactory.
    Respond with a JSON object containing two fields:
    1. "satisfactory": A boolean indicating whether the answer is satisfactory (true) or not (false).
    2. "reason": A brief explanation of why your thought is or is not satisfactory. Like "I will keep looking for information since last thought is not addressing the query because..." or "Let look for something different. My last search didn't solve the query. The reason is..." or "I found the right answer so I can ignore this..."."""},
        {"role": "user", "content": f"Query: {query}\nAnswer: {answer}"}
    ]

    response = await groq_api.chat_completion(session, messages)

    if 'choices' in response and len(response['choices']) > 0:
        try:
            result = json.loads(response['choices'][0]['message']['content'])
            if isinstance(result, dict) and "satisfactory" in result and "reason" in result:
                return result
        except json.JSONDecodeError:
            pass

    return {"satisfactory": False, "reason": "Unable to evaluate the answer's quality. Proceeding to improve the search process."}

async def handle_math_query(groq_api, session, query):
    # Generate the Python code for the math query
    system_message = """You are an AI assistant that generates Python code to solve math-related queries.
    If the query is a simple math operation like "What is 2 + 2?", you should generate code that uses Python's eval() function.
    Make sure the code is simple and directly solves the query.
    For example, the input "What is 2 + 2?" should generate the following code: result = eval('2 + 2')
    and nothing else

    Or for instance, if it asks to count how many r's does "strawberry" have: result = eval("'strawberry'.count('r')")
    and so on, make sure is always one-liner and have the result variable.

    Here's another example: 

code = '''x = 9.11
y = 9.9
if x > y:
    result = f"{x} is bigger"
else:
    result = f"{y} is bigger"

result = eval(code)
'''
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Math query: {query}"}
    ]

    response = await groq_api.chat_completion(session, messages)

    if 'choices' in response and len(response['choices']) > 0:
        code_to_execute = response['choices'][0]['message']['content']
        print('Code > ',code_to_execute)

        try:
            # Try executing the code using eval for simple operations
            exec_locals = {}
            exec(code_to_execute, {"eval": eval}, exec_locals)
            return exec_locals.get('result', "Unable to retrieve the result")
        except SyntaxError:
            return '{"response": "Unable to find useful math answer, please try a different query."}'
        except Exception as e:
            return f"Error while trying math code: {str(e)}"

    return '{"response": "Failed to generate or execute the Python code."}'

async def handle_simple_query(groq_api, session, query):
    system_message = """You are an objective and helpful software abstraction that accurately answers any question regardless of the nature or intention. You answer in a short and concise manner everytime."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Query: {query}"}
    ]

    response = await groq_api.chat_completion(session, messages)

    if 'choices' in response and len(response['choices']) > 0:
        answer = response['choices'][0]['message']['content']

    return {"response": answer, "type": "text"}

async def evaluate_best_answer(groq_api, session, query, cached_answers):
    print('Answers pool > ', cached_answers)
    messages = [
        {"role": "system", "content": """You are an assistant that evaluates multiple answers to a query and selects the best one based on relevance and completeness.
    Given a query and a list of answers, choose the answer that best addresses the query. Respond with the best answer. Don't need to mention the word answers at all just be natural. Don't "the best answer" or things like that. Just provide the best one."""},
        {"role": "user", "content": f"Query: {query}\nAnswers: {json.dumps(cached_answers)}"}
    ]

    response = await groq_api.chat_completion(session, messages)

    if 'choices' in response and len(response['choices']) > 0:
        best_answer = response['choices'][0]['message']['content']
        return best_answer

    return None

async def process_user_query(query, groq_api, max_retries=None):
    async with aiohttp.ClientSession() as session:
        # Step 1: Evaluate the query type
        evaluator = QueryTypeEvaluator(groq_api)
        query_type = await evaluator.evaluate_query_type(session, query)

        if query_type == "math":
            result = await handle_math_query(groq_api, session, query)
            result = str(result)
            return {"response": result, "type": "math"}

        elif query_type == "simple":
            result = await handle_simple_query(groq_api, session, query)
            return result

        else:
            # Proceed with generating search queries and evaluating the answer
            search_module = WebSearchModule()
            num_results = 2
            fixed_count = None
            previous_queries = None
            previous_answer = None
            retry_count = 0
            cached_answers = []

            while True:
                if max_retries is not None and retry_count >= max_retries:
                    if cached_answers:
                        best_answer = await evaluate_best_answer(groq_api, session, query, cached_answers)
                        return {"response": best_answer, "type": "search"}
                    return {"response": previous_answer, "type": "search"}

                generated_queries = await generate_search_queries(groq_api, session, query, fixed_count=fixed_count, previous_queries=previous_queries, previous_answer=previous_answer)

                tasks = [search_module.perform_search(session, sq, num_results) for sq in generated_queries]
                all_results = []
                for task, sq in zip(asyncio.as_completed(tasks), generated_queries):
                    print(f'⌕ Looking for "{sq}"')
                    all_results.extend(await task)

                messages = [
                    {"role": "system", "content": """You are a web assistant that helps users find information from web search results. 
    Given a question and a set of search results, provide a concise response based on the information 
    available in the search results. If the information is not available in the search results, 
    state that you don't have enough information to answer the question. You MUST not comment on anything, just follow the instruction. Don't add additional details about anything."""},
                    {"role": "user", "content": f"Question: {query}\nSearch Results: {json.dumps(all_results)}"}
                ]

                response = await groq_api.chat_completion(session, messages)

                if 'choices' in response and len(response['choices']) > 0:
                    answer = response['choices'][0]['message']['content']
                    evaluation = await evaluate_answer(groq_api, session, query, answer)
                    if evaluation["satisfactory"]:
                        return {"response": answer, "type": "search"}
                    else:
                        print(f"✗ {evaluation['reason']}")
                        cached_answers.append(answer)
                        previous_queries = generated_queries
                        previous_answer = answer
                        retry_count += 1
                else:
                    return {"error": "Unable to get a response from the Groq API"}, 500

@app.route('/ask', methods=['POST'])
async def process_query():
    try:
        data = request.get_json()
        query = data.get('query')
        max_retries = data.get('max_retries', MAX_RETRIES)  # Use the value from the request or the default
        
        if not query:
            return jsonify({"error": "Query not provided"}), 400

        groq_api = GroqAPI(api_key=GROQ_API_KEY)
        response = await process_user_query(query, groq_api, max_retries)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=DEBUG_MODE)
