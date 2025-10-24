"""
Este archivo va a manejar el bug nuevo que llega -> usa LLM para emprolijarlo o agregarle key words para mejor búsqueda
en pinecone 
"""

import logging
import sys
from pathlib import Path
from openai import OpenAI

# Ensure project root is on the path for absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

logger = logging.getLogger(__name__)

def get_bug_query_reformulation_prompt():
    """
    Returns a prompt template for reformulating user bug queries to optimize 
    vector database searches in Pinecone.
    """
    return """
Reformulate the following user query to use technical development and testing terminology, optimized for searching in a bug tracking database.

Objectives:
- Replace colloquial terms with technical development vocabulary
- Add relevant synonyms from the software development domain (e.g., "bug", "defect", "issue", "error", "failure", "crash", "exception")
- Include technical context when relevant (component, module, functionality, behavior)
- Add severity/priority indicators when appropriate (critical, high, medium, low, blocker)
- Include bug types when relevant (frontend, backend, integration, performance, security, race condition, memory leak)
- Return only the reformulated sentence, without comments or explanations
- **If the user query is purely social, conversational, or irrelevant to software bugs, errors, or development issues (e.g., greetings, jokes, personal opinions, weather, sports), respond exactly with: NON-BUG**
- **If the query relates even indirectly to software issues, errors, bugs, crashes, failures, or any development-related problems, treat it as BUG-RELATED and reformulate it**

Examples:
- User: "my app keeps crashing when I click the button"
- Reformulation: "application crash defect when user interaction triggers button click event in frontend component"
- User: "the website is really slow"
- Reformulation: "performance issue defect causing slow response time and poor user experience in web application"
- User: "hello, how are you?"
- Response: NON-BUG

Now reformulate this user query:
<user_question>{user_question}</user_question>
"""

def reformulate_bug_query(user_query: str) -> str:
    """
    Reformulates a user bug query to optimize vector database search.
    
    Args:
        user_query (str): The original user query about a bug
    
    Returns:
        str: Reformulated query optimized for vector search, or "NON-BUG" if not bug-related
    """
    try:
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        prompt_template = get_bug_query_reformulation_prompt()
        formatted_prompt = prompt_template.format(user_question=user_query)

        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )

        if response.choices and response.choices[0].message.content:
            reformulated = response.choices[0].message.content.strip()
            logger.info(f"Query reformulated: '{user_query}' -> '{reformulated}'")
            return reformulated
        else:
            logger.warning("Empty response from LLM, using original query")
            return user_query
            
    except Exception as e:
        logger.error(f"Error reformulating query: {e}")
        return user_query 