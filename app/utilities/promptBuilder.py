# This file is used to interact with database
from app.utilities.dbInteractions import get_activity_type_configuration, get_content_template
from app.utilities.utility import transform_keyvalue_list_to_string
from app import obj_jd_search
from app.utilities.logger import logger
from config import PERSONA_INCLUSION_IN_GENERATE_PROMPT, STYLE_INCLUSION_IN_GENERATE_PROMPT


def construct_LLM_prompt(value: dict, activity_type_config: dict, userData: dict) -> str:
    """
    This method is used to generate the prompt to generate content 
    using an LLM.
    @param payload: dict
    @return: dict
    """
    try:
        # Read the input configuration to generate the content
        config, keywords, themes, purpose, outline, content_template_items, audience, entity, tone, min_words, max_words, link, template, style_description = read_input_configuration(value, activity_type_config, userData)

        persona_prompt_section = ""
        if themes is not None and len(themes) > 0:
            persona_prompt_section = PERSONA_INCLUSION_IN_GENERATE_PROMPT.format(
                themes=themes,
                tone=tone
            )

        if template is None or len(template) == 0:
            # Configure the prompt based on the generate template configured for the content/ activity type
            prompt = config.get('generatePromptTemplate').format(
                activityTypeDescription=activity_type_config.get('activityTypeDescription'),
                topic=purpose,
                keywords=keywords,
                audience=audience,
                includeLink=link,
                min_words=min_words,
                max_words=max_words,
                tone=tone
            )
        else:
            # Configure the prompt based on rewrite template configured for the content/ activity type
            prompt = config.get('rewritePromptTemplate').format(
                activityTypeDescription=activity_type_config.get('activityTypeDescription'),
                template=template,
                keywords=keywords,
                min_words=min_words,
                max_words=max_words,
                tone=tone
            )

        # Flag that no context is available if there is no outline
        context_unavailable = (len(outline) == 0)
        # In case a system default or org specific content template is available, construct as per the template
        if len(content_template_items) > 0:
            outlineMap = dict()
            for outline_item in outline:
                outlineMap[outline_item.get('sectionId')] = outline_item.get('text')

            prompt = prompt + '\r\nFirstly, the ' + activity_type_config.get('activityTypeDescription') + ' should include the following sections:'
            
            # For each section of the template, include key topics that need to be included in the output
            i=1
            for template_item in content_template_items:
                prompt = prompt + '\r\n' + str(i) + '. Section title: ' + (template_item.get('section_description') or template_item.get('section_name'))
                
                if template_item.get('section_id') in outlineMap:
                    if template_item.get('configuration') is not None:
                        # If custom configuration is available for the template item, use it
                        prompt = prompt + '. This section should be in ' + template_item.get('configuration').get('style') + ' style and include the following topics: """' + outlineMap[template_item.get('section_id')] + '"""'
                    else:
                        prompt = prompt + '. This section should include the following topics: """' + outlineMap[template_item.get('section_id')] + '"""'
                else:
                    if template_item.get('configuration') is not None:
                        # If custom configuration is available for the template item, use it
                        prompt = prompt + '. This section should be in ' + template_item.get('configuration').get('style') + ' style.'
                i=i+1

        if keywords is not None and len(keywords) > 0:
            prompt = prompt + '\r\nThis ' + activity_type_config.get('activityTypeDescription') + ' should include the following topics: """' + keywords + '"""'
        if themes is not None and len(themes) > 0:
            prompt = prompt + persona_prompt_section
        
        # Incorporate the presentation style, if available
        if style_description is not None:
            presentation_style_instruction = STYLE_INCLUSION_IN_GENERATE_PROMPT.format(
                style_description=style_description
            )
            prompt = prompt + presentation_style_instruction

        return {
                    'title': "",
                    'description': "",
                    'prompt': prompt,
                    'context_unavailable': context_unavailable
                }

    except Exception as err:
        print(err)
        logger.error(err.args)
        return {"error": err.args or "Something went wrong!", "status": 404}


def construct_sectionwise_LLM_prompt(value: dict, activity_type_config: dict, userData: dict) -> str:
    """
    This method is used to generate the section-wise prompt to generate content 
    using an LLM.
    @param payload: dict
    @return: dict
    """
    try:
        # Read the input configuration to generate the content
        config, keywords, themes, purpose, outline, content_template_items, audience, entity, tone, min_words, max_words, link, template, style_description = read_input_configuration(value, activity_type_config, userData)

        sectionPrompts = []
        # In case a system default or org specific content template is available, construct as per the template
        if len(content_template_items) > 0:
            outlineMap = dict()
            for outline_item in outline:
                outlineMap[outline_item.get('sectionId')] = outline_item.get('text')
            
            # For each section of the template, include key topics that need to be included in the output
            # i=0
            section_min_words = int(min_words/len(content_template_items))
            section_max_words = int(max_words/len(content_template_items))
            for template_item in content_template_items:
                # Construct the prompt with these inputs
                prompt, context_unavailable = construct_section_prompt(activity_type_config, config, keywords, themes, purpose, audience, entity, tone, link, template, outlineMap, section_min_words, section_max_words, template_item, style_description)
                    
                boilerplate_available = False
                # Check if a boilerplate has been provided for the section
                if isinstance(template_item.get('snippet'), list) and len(template_item.get('snippet')) > 0 and len(template_item.get('snippet')[0]) > 0:
                    boilerplate_available = True

                # If a boilerplate has been provided for the section, include it
                sectionPrompts.append({
                    'title': template_item.get('section_name'),
                    'description': template_item.get('section_description'),
                    'prompt': prompt,
                    'context_unavailable': context_unavailable,
                    'boilerplate': template_item.get('snippet')[0] if boilerplate_available else None
                })

        return sectionPrompts

    except Exception as err:
        print(err)
        logger.error(err.args)
        return {"error": err.args or "Something went wrong!", "status": 404}


def read_input_configuration(value, activity_type_config, userData):
    """
    This method is used to read the input configuration for content generation.
    @param value: dict
    @param activity_type_config: dict
    @param userData: dict
    @return: dict
    """
    if value.get('activityType') is None:
        raise ValueError("'activityType' not present!")

        # Either 'purpose' or 'template' (existing content to convert) should be provided
    if (value.get('purpose') is None and value.get('primaryInput') is None) and value.get('template') is None:
        raise ValueError("Either 'purpose', 'primaryInput' (questionnaire) or 'template' (existing content to convert) should be provided!")

        # Some required params to fetch activity_type configuration
    org_id = userData.get('orgId')
    org_code = userData.get('orgCode')
    industry = userData.get('industry')

    config = activity_type_config.get('configuration')

    style_description = None
    presentation_style = value.get('presentationStyle')
    if presentation_style is not None:
        style_description = presentation_style.get('style_description')

        # Fetch key themes for the content to be created
    keywords = '\r\n'.join(value.get('keywords')) \
            if isinstance(value.get('keywords'), list) else value.get('keywords')
        # If the inputs are based on a template use those inputs instead
    if isinstance(value.get('primaryInput'), list):
        keywords = transform_keyvalue_list_to_string(value.get('primaryInput')) 

    themes = '\r\n'.join(value.get('themes')) \
            if isinstance(value.get('themes'), list) else value.get('themes')
        
        # Fetch the purpose. If not passed by user, 
        # get the default configuration for the content/ activity type
    purpose = value.get('purpose')
    outline = []
    if value.get('purpose') is None or len(purpose) == 0:
        purpose = activity_type_config.get('purpose')
        # In case the purpose is provided, lookup the closest content already available in the content repository
    else: 
        if value.get('activityType') == 'DEFAULT_JOB_DESCRIPTION':
            content_search_results = obj_jd_search.process_query(purpose, value.get('keywords'), org_code, industry)
                # Validate that the search returned non-empty documents
            if len(content_search_results) > 0 and len(content_search_results[0].get('documents')) > 0:    
                if len(content_search_results[0].get('documents')[0].get('vectorMatchScore')) > 0 and  content_search_results[0].get('documents')[0].get('vectorMatchScore')[0] > 0.9:
                        outline = content_search_results[0].get('documents')[0].get('docMetadata').get('sections')
        
        # Alternately, in case the user has provided an outline use it
    if len(outline) == 0 and value.get('outline') is not None:
        outline = value.get('outline') \
                if isinstance(value.get('outline'), list) else []


        # Get template corresponding to the content type
    content_template_items = get_content_template(
            content_type=activity_type_config.get('contentType'),
            org_id=org_id
        )

        # Fetch the target audience. If not passed by user, 
        # get the default configuration for the content/ activity type
    audience = ', '.join(value.get('audience')) \
            if isinstance(value.get('audience'), list) else value.get('audience')
    if audience is None or len(audience) == 0:
        audience = activity_type_config.get('audience')

        # Fetch the content creator organization
    entity = value.get('entity')

        # Fetch additional language inputs. If not overwritten by user, fetch defaults for the actvity type
    tone = value.get('tone')
    if isinstance(tone, list):
        tone = '\r\n'.join(tone)
    elif tone is None or len(tone) == 0:
        tone = config.get('tone')

    if value.get('wordLimit') is not None and value.get('wordLimit').get('min') is not None:
        min_words = int(value.get('wordLimit').get('min'))
    else:
        min_words = int(config.get('min_tokens') * 0.75)
        
    if value.get('wordLimit') is not None and value.get('wordLimit').get('max') is not None:
        max_words = int(value.get('wordLimit').get('max'))
    else:
        max_words = int(config.get('max_tokens') * 0.75)

    link = value.get('link')

        # In case a template has been provided for the content, rewrite based on it
        # else generate the content from scratch
    template = value.get('template') or ''

    return config,keywords,themes,purpose,outline,content_template_items,audience,entity,tone,min_words,max_words,link,template,style_description


def construct_section_prompt(activity_type_config, config, keywords, themes, purpose, audience, entity, tone, link, template, outlineMap, section_min_words, section_max_words, template_item, style_description):
    """
    This helper method is used to generate the prompt for individual section.
    @param payload: dict
    @return: dict
    """
    persona_prompt_section = ""
    if themes is not None and len(themes) > 0:
        persona_prompt_section = PERSONA_INCLUSION_IN_GENERATE_PROMPT.format(
            themes=themes,
            tone=tone
        )

    if template is None or len(template) == 0:
                    # Configure the prompt based on the generate template configured for the content/ activity type
        prompt = config.get('generateSectionPromptTemplate').format(
                        activityTypeDescription=activity_type_config.get('activityTypeDescription'),
                        sectionTitle=template_item.get('section_name'),
                        topic=purpose,
                        keywords=keywords,
                        # entity=entity,
                        audience=audience,
                        includeLink=link,
                        min_words=section_min_words,
                        max_words=section_max_words,
                        tone=tone
                    )
    else:
                    # Configure the prompt based on rewrite template configured for the content/ activity type
        prompt = config.get('rewriteSectionPromptTemplate').format(
                        activityTypeDescription=activity_type_config.get('activityTypeDescription'),
                        template=template,
                        keywords=keywords,
                        min_words=section_min_words,
                        max_words=section_max_words,
                        tone=tone
                    )
                
    context_unavailable = True
    # If there is relevant themes available for the section, include it
    if template_item.get('section_id') in outlineMap:
        context_unavailable = False
        prompt = prompt + 'The section should have the title: """' + template_item.get('section_name') + '."""'
        if template_item.get('configuration') is not None:
            # If custom configuration is available for the template item, use it
            prompt = prompt + 'This section should be in ' + template_item.get('configuration').get('style') + ' style and include the following topics: """' + outlineMap[template_item.get('section_id')] + '"""'
        else:
            prompt = prompt + 'This section should include the following topics: """' + outlineMap[template_item.get('section_id')] + '"""'
    else:
        if template_item.get('configuration') is not None:
            # If custom configuration is available for the template item, use it
            prompt = prompt + 'This section should be in ' + template_item.get('configuration').get('style') + ' style.'

    if (keywords is not None and len(keywords) > 0) and (template_item.get('enrichment') is not None and template_item.get('enrichment').get('themes') == True):
        prompt = prompt + '\r\nInclude the following topics: """' + keywords + '"""'
    if themes is not None and len(themes) > 0:
        prompt = prompt + persona_prompt_section
    
    # Incorporate the presentation style, if available
    if style_description is not None:
        presentation_style_instruction = STYLE_INCLUSION_IN_GENERATE_PROMPT.format(
            style_description=style_description
        )
        prompt = prompt + presentation_style_instruction

    return prompt,context_unavailable
