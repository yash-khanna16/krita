import openai
import json
import re
import google.generativeai as palm
import os
from uuid import uuid4
from ast import literal_eval
import numpy as np
from numpy import argmax
from datetime import date, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from app.utilities.logger import logger
import google
from app.utilities.dbInteractions import get_campaign_construct
from config import OPENAI_API_KEY, EMBEDDING_MODEL, QDRANT_URL, \
    QDRANT_API_KEY, QDRANT_CONTENT_COLLECTION_NAME, CONTENT_CLASSIFICATION_METADATA_PROMPT, \
        CONTENT_EXTRACTION_MODEL, CONTENT_FEATURES_FOR_KEYWORDS, \
            CAMPAIGN_SUGGESTION_METADATA_PROMPT, CRITERIA_MATCH_THRESHOLDS, \
                BRAND_CRITERIA_MATCH_THRESHOLD, BRAND_NAME_MISSING_PENALTY, CONTENT_RESULTS_LIMIT, \
                TOP_PERSONA_EXTRACTION_PROMPT
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from google.generativeai.types import model_types

openai.api_key = OPENAI_API_KEY

# persona extractor

def validate_data(data):
    if 'data' not in data or 'SocialPosts' not in data['data']:
        raise ValueError("No content provided.")

def create_messages(content, input_data):
    messages = [
        {"role": "system", "content": "You are a talent marketing content writer."},
        {"role": "user", "content": "Extract a persona from the following text and provide the extracted information in the specified JSON format. Please include the persona, themes, and tone."},
        {"role": "user", "content": input_data + "\n" + "\n".join(content)}
    ]
    return messages

def extract_persona_for_batch(input_data: str, multiple: bool):
    """
    This function extracts the persona for a batch of input content.
    @param content: input_data
    :return: dict
    """
    try:
        if multiple:
            count = 'up to two'
        else:
            count = 'one'
        
        messages = [
                    {"role": "system", "content": "You are a talent marketing content writer."},
                    {
                        "role": "user",
                        "content": TOP_PERSONA_EXTRACTION_PROMPT.replace('{{count}}', count).replace('{{posts}}', '"""' + input_data + '"""')
                    }
                ]

        response = make_openai_request(messages)
        return response
    
    except Exception as err:
        logger.error('Error while extracting persona from content text using LLM API:', err.args)
        raise Exception(err)

def make_openai_request(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0.4,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response

def make_paLM_request(Messages):
    print("making request")
    response= google.generativeai.chat(
        model='models/chat-bison-001',
        # context= messages,
        temperature=0.3,
        top_p=1,
        messages=Messages
    )
    return response

def transform_json(original_json):
    # fomatting paLM response
    data_keywords = original_json.get('data', {}).get('keywords', [])
    formatted_keywords = [
        {"keyword": keyword["keyword"], "metadata": keyword.get("metadata", {})}
        for keyword in data_keywords
    ]
    formatted_output = {"data": {"keywords": formatted_keywords}}

    return formatted_output

def parse_generated_output(generated_output):
    try:
        parsed_output = json.loads(generated_output)
        return parsed_output
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in the generated output.")
    

def str_to_dict(content: str) -> dict:
    """
    This function converts the string to a dictionary.
    @param content: str
    :return: dict
    """
    try:
        content_dict = literal_eval(content)
        return content_dict
    except Exception as err:
        logger.error('Error while converting the content to the valid json:', err.args)
        return {'error': 'Cannot convert the jd metadata into a valid json',
                'content': content,
                'status': 400}


def extract_metadata_from_content(content: str) -> dict:
    """
    This function takes the content text and calls the openAI
    GPT-3 to extract the metadata from it.
    @param content: str
    :return: dict
    """
    try:
        content = content.strip('\n').strip().strip('\n')
        response = openai.ChatCompletion.create(
            model=CONTENT_EXTRACTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an Employer Talent Marketing content analyzer."
                },
                {
                    "role": "user",
                    "content": CONTENT_CLASSIFICATION_METADATA_PROMPT.replace('{{}}', '"""' + content + '"""')
                }
            ],
            temperature=0.4,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Convert response text to a valid json
        metadata_dict = str_to_dict(response['choices'][0]['message']['content'])
        return metadata_dict
    except Exception as err:
        logger.error('Error while extracting metadata from content text using GPT-3:', err.args)
        raise Exception(err)


def get_campaign_suggestion_from_content(content: str, topics: list, campaign_construct: dict, org_name: str) -> dict:
    """
    This function takes the content text and calls the openAI
    GPT-3 to get a campaign suggestion from it based on org-defined campaign strategy.
    @param content: str
    :return: dict
    """
    try:
        content = content.strip('\n').strip().strip('\n')

        persona_name = 'Talent Marketer'
        themes = []
        tone = []
        if campaign_construct is not None and campaign_construct.get('configuration') is not None:
            configuration = campaign_construct.get('configuration')
            # Get the brand persona to determine the brand themes and tone to generate campaign suggestion
            brand_persona = configuration.get('brand_persona')[0]

            persona_name = brand_persona.get('persona_name')
            themes = brand_persona.get('configuration').get('themes') if brand_persona.get('configuration') is not None else []
            tone = brand_persona.get('configuration').get('language_and_tone') if brand_persona.get('configuration') is not None else []
        
        # Include the organization name in themes to include
        themes.append(org_name)

        response = openai.ChatCompletion.create(
            model=CONTENT_EXTRACTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an Employer Talent Marketing content writer."
                },
                {
                    "role": "user",
                    "content": CAMPAIGN_SUGGESTION_METADATA_PROMPT
                        .replace('{{topics}}', ' ,'.join(topics).strip())
                        .replace('{{persona_name}}', persona_name)
                        .replace('{{themes}}', ' ,'.join(themes))
                        .replace('{{tone}}', ' ,'.join(tone))
                }
            ],
            temperature=0.6,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Convert response text to a valid json
        metadata_dict = str_to_dict(response['choices'][0]['message']['content'])
        return metadata_dict
    except Exception as err:
        logger.error('Error while extracting metadata from content text using GPT-3:', err.args)
        raise Exception(err)


def prepare_data_for_vector_db(content_text: str, content_metadata: dict, metadata_dict: dict, campaign_metadata_dict: dict, content_type: str, org_code: str,
                               industry: str, campaign_construct_for_event_type: dict) -> dict:
    """
    This function transforms the extracted metadata to make
    it suitable to be inserted into the vector db.
    @param content_text: str
    @param content_metadata: dict
    @param metadata_dict: dict
    @param campaign_metadata_dict: dict
    @param content_type: str
    @param org_code: str
    @param industry: str
    :return: dict
    """
    try:
        # Consolidate all the searchable text fields into keywords for vectorization
        keywords = ''
        for key in content_metadata.keys():
            if key in CONTENT_FEATURES_FOR_KEYWORDS:
                keywords += content_metadata[key] + '\n'
        topic = content_text.strip()
        keywords = keywords.strip()
        
        # Include the campaign construct used to generate the AI suggestion, if available
        if campaign_construct_for_event_type is not None and campaign_construct_for_event_type.get('configuration') is not None:
            campaign_metadata_dict['configuration'] = campaign_construct_for_event_type.get('configuration')
        else:
            campaign_metadata_dict['configuration'] = {}

        # Update derived details in the metadata
        doc_metadata = {}
        doc_metadata['contentMetadata'] = content_metadata
        doc_metadata['extractedSynopsis'] = metadata_dict
        doc_metadata['suggestedCampaign'] = campaign_metadata_dict

        data_dict = {
            'uuid': str(uuid4()),
            "sourceId": content_metadata.get('url'),
            'contentType': content_type,
            'orgCode': org_code,
            'industry': industry,
            'topic': topic,
            'keywords': keywords,
            'docMetadata': doc_metadata
        }

        return data_dict

    except Exception as err:
        logger.error('Error while preparing the content for vector db:', err.args)
        raise Exception(err)


def encode_query(query: str) -> list:
    """
    This function is used to encode a query into
    vectors using openai model.
    :param query: str
    :return: List
    """
    try:
        # Convert the text to embeddings
        response = openai.Embedding.create(
            input=query,
            model=EMBEDDING_MODEL
        )
        embeddings = response['data'][0]['embedding']
        return embeddings
    except Exception as err:
        logger.error('Error while converting text to embeddings:', str(err))
        raise Exception(err)


def insert_vectors_into_db(data_dict: dict) -> str:
    """
    This function is used to insert the content into the
    vector db using Qdrant.
    @param data_dict: dict
    :return: str
    """
    try:
        # Initializing the Qdrant client
        client = QdrantClient(
            url=QDRANT_URL,
            prefer_grpc=True,
            api_key=QDRANT_API_KEY,
        )

        # Collection dimensions to create if it does not exist, keep commented except on first use or to reset
        # client.recreate_collection(
        #     collection_name=QDRANT_CONTENT_COLLECTION_NAME,
        #     vectors_config={
        #         "topic": models.VectorParams(size=1536, distance=models.Distance.COSINE),
        #         "keywords": models.VectorParams(size=1536, distance=models.Distance.COSINE),
        #     }        
        # )
                
        client.upsert(
            collection_name=QDRANT_CONTENT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=data_dict['uuid'],
                    vector={
                        "topic": data_dict["topic_vector"],
                        "keywords": data_dict["keywords_vector"],
                    },
                    payload={
                        "sourceId": data_dict['sourceId'],
                        "contentType": data_dict['contentType'],
                        "orgCode": data_dict["orgCode"],
                        "industry": data_dict["industry"],
                        "topic": data_dict['topic'],
                        "keywords": data_dict['keywords'],
                        "docMetadata": data_dict["docMetadata"]
                    }
                )
            ]
        )
        document_uuid = data_dict['uuid']
        return document_uuid
    except Exception as err:
        logger.error('Error while inserting the content into the vector DB:', err.args)
        raise Exception(err)


def process_content(content_text: str, content_metadata: dict, content_type: str, org_code: str, org_name:str, industry: str) -> dict:
    """
    This function processes content such as social post, optionally extracts any additional information
    and load the vectors in to the vector database.
    @param content_text: str
    @param content_metadata: dict
    @param content_type: str
    @param org_code: str
    @param industry: str
    :return: str
    """
    try:
        # Extract metadata from jd text
        metadata_dict = extract_metadata_from_content(content_text)
        if 'error' in metadata_dict:
            return metadata_dict
        
        # Validate whether the content item meets the relevance criteria for the organization
        relevant_content = metadata_dict.get('relevantforCandidatesOrEmployees') != 'Low' and metadata_dict.get('eventType') != 'Other'

        # If relevant then only proceed to extract campaign suggestion and store the content
        if relevant_content == True:
            # Get campaign construct for the organization
            campaign_construct = get_campaign_construct(
                org_id=org_code
            )
            
            campaign_construct_for_event_type = None
            # Get the campaign strategy for the event type identified
            for campaign_construct_item in campaign_construct:
                if campaign_construct_item.get('event_type_code') == metadata_dict.get('eventType'):
                    campaign_construct_for_event_type = campaign_construct_item
                    break
            
            # Get the campaign suggestion based on org-defined campaign strategy
            campaign_metadata_dict = get_campaign_suggestion_from_content(content_text, metadata_dict.get('topics'), campaign_construct_for_event_type, org_name)

            # Prepare data to be inserted in vector DB
            data_dict = prepare_data_for_vector_db(content_text, content_metadata, metadata_dict, campaign_metadata_dict, content_type, org_code, industry, campaign_construct_for_event_type)

            # Convert the relevant columns into the vector
            data_dict['topic_vector'] = encode_query(data_dict['topic'])
            data_dict['keywords_vector'] = encode_query(data_dict['keywords'])

            # Get the match score against the brand criteria
            criteria_match_expected_score, criteria_match_actual_score = get_brand_criteria_match_score(campaign_construct_for_event_type, data_dict, org_name)

            data_dict["docMetadata"]['criteriaMatchExpectedScore'] = criteria_match_expected_score
            data_dict["docMetadata"]['criteriaMatchActualScore'] = criteria_match_actual_score

            # Add the content publish/ discovery date
            data_dict["docMetadata"]['content_date'] = int(date.today().strftime("%Y%m%d"))

            # Validate that the brand criteria does not deviate beyond the pre-configured threshold
            if criteria_match_expected_score - criteria_match_actual_score <= BRAND_CRITERIA_MATCH_THRESHOLD:
                # Insert the vectors into the vector db
                content_uuid = insert_vectors_into_db(data_dict)
                return {
                    'UUID': content_uuid
                }
            else:
                return {
                    'err': 'BRAND_CRITERIA_MATCH_REJECTION'
                }
        else:
            return {
                'err': 'RELEVANCE_CRITERIA_REJECTION'
            }

    except Exception as err:
        logger.error('Error while processing content:', err.args)
        raise Exception(err)


def get_brand_criteria_match_score(campaign_construct_for_event_type: dict, data_dict: dict, org_name: str):
    """
    This method is used to perform the vector search
    based on the given query.
    :param campaign_construct_for_event_type: dict
    :param data_dict: dict
    :return: multiple
    """
    criteria_themes = []
    criteria_match_expected = "Medium"
    criteria_match_actual_score = CRITERIA_MATCH_THRESHOLDS['Low']

    # Check if the campaign strategy is available for the identified event type 
    if campaign_construct_for_event_type is not None and campaign_construct_for_event_type.get('search_criteria') is not None:
        search_criteria = campaign_construct_for_event_type.get('search_criteria')
        criteria_themes = search_criteria.get('themes') if search_criteria.get('themes') is not None else []
        criteria_match_expected = search_criteria.get('match')  if search_criteria.get('match') is not None else 'Medium'

        # Include the organization name in the criteria themes to match
        criteria_themes_vector = encode_query(org_name + '\n' + ' ,'.join(criteria_themes))

        # Calculate the cosine similarity to determine the degree of match with the brand's search criteria
        topic_criteria_match = cosine_similarity(np.array(data_dict.get('topic_vector')).reshape(1, -1),
                                                            np.array(criteria_themes_vector).reshape(1, -1))[0][0]
        keywords_criteria_match = cosine_similarity(np.array(data_dict.get('keywords_vector')).reshape(1, -1),
                                                            np.array(criteria_themes_vector).reshape(1, -1))[0][0]
        criteria_match_actual_score = round((0.75 * topic_criteria_match) + (0.25 * keywords_criteria_match), 1)

        # If the organization name is completely missing, apply a penalty to reduce the score
        includesOrgName =  re.search(org_name, data_dict['keywords'], flags=re.IGNORECASE)
        if includesOrgName is None:
            criteria_match_actual_score = criteria_match_actual_score - BRAND_NAME_MISSING_PENALTY
    
    criteria_match_expected_score = CRITERIA_MATCH_THRESHOLDS.get(criteria_match_expected)

    return criteria_match_expected_score, criteria_match_actual_score


def process_query(topic_query: str, keywords_query: list,
                    org_code: str, industry: str) -> list:
    """
    This method is used to perform the vector search
    based on the given query.
    :param topic_query: str
    :param keywords_query: list of keywords
    :param org_code: str
    :param industry: str
    :return: list
    """
    try:
        # prepare queries
        keywords_query = '\n'.join(keywords_query)

        # Encode the queries to vectors
        topic_query_vector = encode_query(topic_query)

        # Initializing the Qdrant client
        client = QdrantClient(
            url=QDRANT_URL,
            prefer_grpc=True,
            api_key=QDRANT_API_KEY,
        )

        results_limit = CONTENT_RESULTS_LIMIT
        content_date_threshold = int((date.today() - timedelta(days=7)).strftime("%Y%m%d"))
        
        # Perform a vector search from the vector DB on topic
        topic_results = client.search(
            collection_name=QDRANT_CONTENT_COLLECTION_NAME,
            query_vector=('topic', topic_query_vector),
            query_filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key="orgCode",
                        match=models.MatchValue(value=org_code),
                    ),
                    models.FieldCondition(
                        key="content_date",
                        range=models.Range(
                            gt=None,
                            gte=content_date_threshold,
                            lt=None,
                            lte=None,
                        ),
                    ),
                ]
            ),
            limit=results_limit,
            with_vectors=False,
            with_payload=True,
        )

        if len(topic_results) == 0:
            return [
                {
                    "documents": []
                }
            ]

        output_document_list = []

        # Iterate through the primary vector results to construct the output
        for i in range(len(topic_results)):
            output_document_list.append(
                {
                    "documentId": topic_results[i].id,
                    "documentRelevanceScore": topic_results[i].score,
                    "docMetadata": {
                        'sourceId': topic_results[i].payload['sourceId'],
                        'industry': topic_results[i].payload['industry'],
                        'orgCode': topic_results[i].payload['orgCode'],
                        "topic": topic_results[i].payload['topic'],
                        "details": topic_results[i].payload['docMetadata']
                    }
                }
            )

        response = [
            {
                "documents": output_document_list
            }
        ]

        return response

    except Exception as err:
        logger.error('Error while processing the search query:', str(err))
        raise Exception(err)
