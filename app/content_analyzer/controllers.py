import json
import asyncio
from flask import (
    Blueprint, jsonify, Response, request
)
from os import remove
from config import SET_PERSONA_CONTEXT, TOP_PERSONA_EXTRACTION_PROMPT, PERSONA_EXTRACTION_BATCH_SIZE, CLUSTERING_INPUT_MIN_COUNT,TOP_KEYWORDS_PROMPT
from config import OPENAI_API_KEY
from config import GOOGLE_API_KEY
from app.utilities import responseHandler
from app.utilities.constants import ErrorMessages, SuccessMessages
from app.content_analyzer.services import process_content, process_query, extract_persona_for_batch, make_paLM_request, transform_json
from app.persona.persona_clustering import handle_clustering
from app.utilities.logger import logger
from app.utilities.utility import str_to_dict
from app.openai.services import OpenAIService
from flask import Flask, request, jsonify
import openai
from app.content_analyzer.services import validate_data, create_messages, make_openai_request, parse_generated_output
import google.generativeai as palm
import os
import codecs 

# palm.configure(api_key=os.environ['AIzaSyCE_vA2MfKYh88c5Y68lWa7BE2r7ax6604']) #    11346a42655da87337a73119fb0ee90e1d32c559
palm.api_key = GOOGLE_API_KEY
mod_content_analyzer = Blueprint("content_analyzer", __name__, url_prefix='/content')
openai.api_key = OPENAI_API_KEY
@mod_content_analyzer.errorhandler(404)
def bad_request(error="Not Found"):
    return jsonify(error=error), 404


@mod_content_analyzer.route("/health", methods=['GET'])
def health() -> Response:
    """
    This method is used for health
    check for auth blueprint.
    @return: JSON
    """
    return responseHandler.success_response(
        SuccessMessages.HEALTH_CHECK_DONE.value,
        200
    )


@mod_content_analyzer.route("/extract_persona", methods=['POST'])
def extract_persona() -> Response:
    try:
        data = request.get_json()
        # validate_data(data)

        content = data['data']['SocialPosts']['innerText']
        input_data = "\n".join(content)

        # Extract the persona using LLM
        response = extract_persona_for_batch(input_data, False)
        generated_output = response['choices'][0]['message']['content']

        parsed_output = parse_generated_output(generated_output)

        if isinstance(parsed_output, list) and len(parsed_output) > 0:
            return jsonify({"data": {"SocialPosts": parsed_output[0]}, "status": 200}), 200
        else:
            return jsonify({"data": {"SocialPosts": {}}, "status": 204}), 204

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@mod_content_analyzer.route("/extract_top_persona", methods=['POST'])
def extract_top_persona() -> Response:
    """
    This method is used to extract the top persona from content such as social media posts.
    @return: JSON
    """
    try:
        data = request.get_json()
        # validate_data(data)

        content = data['data']['content']

        # Split content into batches of 20-25 posts each or lesser, if the overall number of posts is less
        if len(content) < PERSONA_EXTRACTION_BATCH_SIZE * 2:
            batch_size = int(PERSONA_EXTRACTION_BATCH_SIZE / 2)
        else:
            batch_size = PERSONA_EXTRACTION_BATCH_SIZE
        batched_content = [content[i:i + batch_size] for i in range(0, len(content), batch_size)]

        all_personas = []
        for batch in batched_content:
            input_data = "\n".join(batch)
            
            # Extract persona for the batch and append to the list
            response = extract_persona_for_batch(input_data, True)
            generated_output = str_to_dict(response['choices'][0]['message']['content'])
            all_personas.extend(generated_output)

        # If more personas than the threshold are extracted, cluster them
        if len(all_personas) > CLUSTERING_INPUT_MIN_COUNT:
            clusters_map, max_silhouette_score = handle_clustering(all_personas)
            
            # Get a unique persona from each distinct cluster
            top_personas = extract_distinct_persona_from_clusters(clusters_map)
        else:
            top_personas = all_personas
        
        # Return the top personas
        return jsonify({"data": {"content": top_personas}, "status": 200}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


def extract_distinct_persona_from_clusters(cluster_data):
    """
    This method is used to extract unique persona from each distinct cluster
    @return: list
    """
    top_personas = []
    for cluster_id, personas in cluster_data.items():
        if personas:
            top_personas.append(personas[0])  
    return top_personas


@mod_content_analyzer.route("/insert_content_into_db", methods=['POST'])
def insert_content_into_db() -> Response:
    """
    This method is used to parse content such as social media posts as plain text/ html,
    generate a campaign suggestion based on it, and insert it into the vector db.
    @return: JSON
    """
    try:
        api_request_data = json.loads(request.data)
        data = api_request_data['data']

        content_text = data['Content'].get('innerText')
        content_metadata = data['Content'].get('metadata')
        content_type = data['contentType']
        org_code = data['orgCode']
        org_name = data['orgName']
        industry = data['industry']

        # Process the content
        content_status = process_content(
            content_text=content_text,
            content_metadata=content_metadata,
            content_type=content_type,
            org_code=org_code,
            org_name=org_name,
            industry=industry
        )

        if 'error' in content_status:
            return responseHandler.failure_response(
                content_status,
                400
            )
        else:
            success_response = {
                'Content': [
                    {
                        'documents': [
                            content_status
                        ]
                    }
                ]
            }
            return responseHandler.success_response(
                success_response,
                200
            )
    except Exception as err:
        logger.error(err.args)
        return responseHandler.failure_response(
            ErrorMessages.SOMETHING_WENT_WRONG.value,
            500
        )


@mod_content_analyzer.route("/search", methods=['POST'])
def semantic_search() -> Response:
    """
    This method is used for vector search on the content insight
    based on the given query.
    @return: JSON
    """
    try:

        api_request_data = json.loads(request.data)
        data = api_request_data['data']

        org_code = api_request_data['orgCode']
        industry = api_request_data['industry']
        response = {}
        for attribute in data.keys():
            res = process_query(data[attribute]['topic'], data[attribute]['keywords'], org_code, industry)
            response[attribute] = res

        return responseHandler.success_response(
            response,
            200
        )
    except Exception as err:
        print('err =>', err)
        logger.error(err.args)
        return responseHandler.failure_response(
            ErrorMessages.SOMETHING_WENT_WRONG.value,
            500
        )

@mod_content_analyzer.route("/fetch_trending_keywords", methods=['POST'])
def extract_keywords() -> Response:
    try:
        data = request.json['data']['query']
        
        input_data = " ".join(data)
        messages = [
            {"author": "1", "content": "You are a Search Engine Optimization tool."},
            {
                "author": "0",
                "content": TOP_KEYWORDS_PROMPT.replace('{{data}}', input_data)
            }
        ]
        
        response = make_paLM_request(messages)
        print(response.last)
        try:
            response_content = response.last
            cleaned_content = response_content.strip('`').replace('json', '')
            
            if cleaned_content:
                original_json = json.loads(cleaned_content)
                transformed_json = transform_json(original_json)
                return jsonify(transformed_json)
            else:
                return jsonify({"error": "Empty response content"}), 500
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Error decoding JSON: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

