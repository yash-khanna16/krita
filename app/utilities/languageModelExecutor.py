import time
# This file is used to interact with database
from app.openai.gpt3 import generate_response
from app.utilities.logger import logger


async def generate_content(title: str, description: str, prompt: str, context_unavailable: bool, boilerplate: str, config: dict, max_tokens: int, content_type: str) -> dict:
    """
    This method is used to generate the prompt to generate content 
    using an LLM.
    @param payload: dict
    @return: dict
    """
    try:
        model_response = dict()

        # If a boilerplate has been provided, use it as-is
        if boilerplate is not None:
            model_response = {
                "title": title,
                "description": description,
                "alternatives": [boilerplate],
                "params_used": { "boilerplate": boilerplate },
                "status": True,
                "execution_params": {},
                "content_type": content_type
            }
            return model_response

        # Call the large language model
        start_time = round(time.time()*1000)
        retry_count = 0
        while retry_count < 3:
            #print('retry_count: ', retry_count)
            gpt3_response = await generate_response(
                model=config.get('model') or "gpt-3.5-turbo-0613", # text-davinci-003
                prompt=prompt,
                temperature=(config.get('no_context_temperature') if context_unavailable else config.get('temperature')) or 0.7,
                max_tokens=max_tokens or 512,
                top_p=config.get('top_p') or 1,
                n=config.get('res_num') or 1,
                model_type=config.get('model_type') or 'Chat'
            )
            retry_count = retry_count + 1
            #print("gpt3_response=> ", gpt3_response)
            if gpt3_response.get('error') or gpt3_response.get('error_message'):
                if retry_count < 3:
                    continue
                else:
                    model_response = { 
                        "err": 'Error returned by LLM when generating content: ' + (gpt3_response.get('error') or gpt3_response.get('error_message')) 
                    }
                    break
            else:
                end_time = round(time.time()*1000) - start_time
        
                model_response = {
                    "title": title,
                    "description": description,
                    "alternatives": gpt3_response.get('data').get('choices'),
                    "params_used": gpt3_response.get('params_used'),
                    "status": gpt3_response.get('status'),
                    "execution_params": { "start_time": int(start_time/1000), "elapsed_time": end_time },
                    "content_type": content_type
                }
                break

        return model_response

    except Exception as err:
        print(err)
        logger.error(err.args)
        return {"error": err.args or "Something went wrong!", "status": 404}
