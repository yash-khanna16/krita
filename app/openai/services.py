import time
import asyncio
import re
import traceback

from app.openai.gpt3 import generate_response
from app.utilities.dbInteractions import get_activity_type_configuration, get_content_template
from app.utilities.promptBuilder import construct_LLM_prompt, construct_sectionwise_LLM_prompt
from app.utilities.languageModelExecutor import generate_content
from config import CONTENT_GENERATION_TOKEN_BUFFER
from app.utilities.logger import logger


class OpenAIService(object):
    """
    Class method for all large language model (ex. OpenAI GPT-3) services
    """

    @classmethod
    def generate_content_service(cls, payload: dict, userData: dict) -> dict:
        """
        This method is used to generate the content 
        using OpenAI GPT-3 API.
        @param payload: dict
        @return: dict
        """
        try:
            res_obj = dict()
            for key, value in payload.items():
                org_id = userData.get('orgId')
                # Get configuration corresponding to activity type
                # Some required params to fetch activity_type configuration
                activity_type = value.get('activityType')
                implementation_key = value.get('implementationKey')
                activity_type_config = get_activity_type_configuration(
                    activity_type=activity_type,
                    implementation_key=implementation_key or "GPT3",
                    org_id=org_id
                )

                if activity_type_config.get('error'):
                    return activity_type_config

                config = activity_type_config.get('configuration')

                if value.get('activityType') is None:
                    raise ValueError("'activityType' not present!")

                if value.get('wordLimit') is not None and value.get('wordLimit').get('max') is not None:
                    max_tokens = int(int(value.get('wordLimit').get('max')) * 1.33)
                else:
                    max_tokens = int(config.get('max_tokens'))
                

                # Construct the prompt
                section_prompts = construct_sectionwise_LLM_prompt(value, activity_type_config, userData)
                #print("section_prompts=> ", section_prompts)
                if section_prompts is None or len(section_prompts) == 0:
                    section_prompts = [construct_LLM_prompt(value, activity_type_config, userData)]
                    # res_obj[key] = generate_content(prompt, config, max_tokens, activity_type_config.get('contentType'))
                    # In case the content is to be generated in one go, augment the max token length
                    # with a buffer to account for the LLM exceeding the specified word count, to ensure the output does not end abruptly
                    max_tokens = int(max_tokens * CONTENT_GENERATION_TOKEN_BUFFER)

                else:
                    # In case the segments have to be generated section-wise, decrease the max token length proportionately
                    # with a buffer to account for the LLM exceeding the specified word count, to ensure the output does not end abruptly
                    max_tokens = int(max_tokens * CONTENT_GENERATION_TOKEN_BUFFER/ len(section_prompts))

                # Call the large language model
                # res_obj[key] = generate_content(prompt, config, max_tokens, activity_type_config.get('contentType'))
                generated_content = ""
                params_used = []
                generated_segments = asyncio.run(get_content_segments_async(section_prompts, config, max_tokens, activity_type_config.get('contentType')))
                
                # Process and consolidate the output for each section
                for section_output in generated_segments:
                    if section_output.get('err') is None:
                        section_text = section_output.get('alternatives')[0]
                        section_title = section_output.get('title')
                        section_description = section_output.get('description') or ''
                        if section_text is not None and section_text != "":
                            # If the section title is not already present in the generated content, prefix it
                            includesTitle =  re.search(section_title, section_text, flags=re.IGNORECASE)

                            if includesTitle is not None and len(section_title) > 0:
                                # If present, substitute with the org specific title. Include any additional suffix characters after the title
                                extra_char = '(' + '|'.join([':', ' :', '', ' ', '\n']) + ')'
                                section_title_with_extra_char = section_title + extra_char 
                                pattern = re.compile(section_title_with_extra_char, re.IGNORECASE)
                                section_text = pattern.sub(section_description  + '\n', section_text)
                            else:
                                generated_content = generated_content + section_description + '\n'

                            # If any entity placeholder is present, substitute with the org name
                            org_name_proxies = '(' + '|'.join(['{ORG}','{{ORG}}','\[Company Name\]']) + ')'
                            if value.get('entity') and value.get('entity') != "":
                                pattern = re.compile(org_name_proxies, re.IGNORECASE)
                                section_text = pattern.sub(value.get('entity'), section_text)

                            # Remove double new lines if any, and replace with a single new line
                            section_text = section_text.replace('\n\n', '\n').replace('"""', '')
                            generated_content = generated_content + section_text + '\n\n'
                        params_used.append({ **section_output.get('params_used'), **section_output.get('execution_params') })
                    else:
                        print('Content section error =>', section_output.get('err'))
                
                res_obj[key] = {
                    "alternatives": [{ 
                        "text": generated_content
                        }],
                    "params_used": params_used,
                    "content_type": activity_type_config.get('contentType')
                }
                #print('generated_content=>', generated_content)
                # for section_prompt in section_prompts:
                #     section_output = generate_content(section_prompt, config, max_tokens, activity_type_config.get('contentType'))
                #     print('section_output: ', section_output)
                #     generated_content = generated_content + (section_output.get('alternatives')[0]).get('text')


            return res_obj

        except Exception as err:
            print(err)
            traceback.print_exc()
            logger.error(err.args)
            return {"error": err.args or "Something went wrong!", "status": 404}


async def get_content_segments_async(section_prompts, config: dict, max_tokens: int, content_type: str):
    """
    This method is used to execute the generative AI calls concurrently.
    @param section_prompt: dict
    @param config: dict
    @param max_tokens: int
    @param content_type: str
    @return: dict
    """
    # Create individual tasks to run concurrently for each section
    tasks = [asyncio.create_task(generate_content(section_prompt.get('title'), section_prompt.get('description'), section_prompt.get('prompt'), section_prompt.get('context_unavailable'), section_prompt.get('boilerplate'), config, max_tokens, content_type)) for section_prompt in section_prompts]

    # Excute generation of all the segments concurrently
    generated_segments = await asyncio.gather(*tasks)

    return generated_segments

