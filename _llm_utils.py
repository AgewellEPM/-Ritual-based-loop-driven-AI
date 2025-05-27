import logging
import os
from openai import OpenAI
import google.generativeai as genai
# import anthropic # Uncomment if you use Anthropic Claude
# from groq import Groq # Uncomment if you use Groq for Llama/Mixtral
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log, RetryError

logger = logging.getLogger(__name__)

# Configure Google Gemini (assuming API key is set in .env)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize clients (globally or passed around)
# For demonstration, we'll pass client instances, but you'd initialize them here.
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# gemini_client = genai.GenerativeModel('gemini-1.5-flash')
# anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.DEBUG))
def _call_llm_api_core(llm_client_instance, model, messages, temperature, max_tokens, response_format=None, **kwargs):
    """
    Internal function for direct LLM API call, decorated with tenacity.
    Supports OpenAI and Gemini, extensible to others.
    """
    try:
        # OpenAI API calls
        if isinstance(llm_client_instance, OpenAI):
            if response_format:
                return llm_client_instance.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs
                )
            else:
                return llm_client_instance.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
        # Google Gemini API calls
        elif isinstance(llm_client_instance, genai.GenerativeModel):
            # Gemini has different message structure and response format handling
            gemini_messages = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in messages]
            # Gemini response_format is implicit or handled via specific tools/schemas
            # For JSON output, we often rely on prompt instruction and then parse
            response = llm_client_instance.generate_content(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    # response_mime_type="application/json" # This can be used if available and desired
                ),
                **kwargs
            )
            # Gemini's response structure is different, we need to adapt it
            # Simulate OpenAI's choices[0].message.content
            return type('obj', (object,), {
                'choices': [
                    type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': response.text
                        })
                    })
                ]
            })
        # Add other LLM clients (Anthropic, Groq, etc.) here
        # elif isinstance(llm_client_instance, anthropic.Anthropic):
        #     return llm_client_instance.messages.create(...)
        # elif isinstance(llm_client_instance, Groq):
        #     return llm_client_instance.chat.completions.create(...)
        else:
            raise ValueError(f"Unsupported LLM client instance type: {type(llm_client_instance)}")

    except Exception as e:
        logger.error(f"LLM API call failed: {e}", exc_info=True)
        raise

def call_llm_with_retry(llm_client_instance, model, messages, temperature, max_tokens, response_format=None, **kwargs):
    """
    Wrapper for LLM API calls with retry logic, handling RetryError explicitly.
    Accepts client_instance as an argument.
    """
    try:
        return _call_llm_api_core(llm_client_instance, model, messages, temperature, max_tokens, response_format, **kwargs)
    except RetryError as e:
        logger.error(f"LLM API call failed after multiple retries: {e}")
        raise ConnectionError("Failed to connect to LLM API after multiple retries.") from e
