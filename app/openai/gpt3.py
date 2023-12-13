"""Contains all the methods corresponding to OPENAI GPT-3 APIs"""
import openai
import os

from app.utilities.logger import logger
from config import OPENAI_API_KEY

logger.info("Setting up OPENAI API Key")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")


# Getting the response from GPT-3 model
async def generate_response(
        model: str = "gpt-3.5-turbo-0613",
        prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1,
        n: int = 1,
        model_type: str = 'Chat'
):
    try:
        logger.info("Calling the model")
        # Send an API request and get a response
        if model_type == 'Chat':
            # Use chat completion mode
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert Job Description writer."},
                    {"role": "user", "content": prompt }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n
            )
            choices = { "choices": [response['choices'][0]['message']['content']] }
        else:
            # Use completion mode
            response = await openai.Completion.acreate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n
            )
            choices = { "choices": [response['choices'][0]['text']] }

        logger.info("Response obtained from model")

        # Params passed to the model
        params_used = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "res_num": n
        }

        return {
            "data": choices,
            "params_used": params_used,
            "status": 200
        }

    except Exception as err:
        logger.error(err.args)
        return {"error": "Something went wrong! Please try again after some time.", "status": 404}
