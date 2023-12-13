import json

from flask import (
    Blueprint, jsonify, Response, request
)
from os import remove
from app.jd_vector_search.jd_vectordb_upsert import process_jd, read_jd_file
from app.utilities import responseHandler
from app.utilities.constants import ErrorMessages, SuccessMessages
from app.utilities.logger import logger
from app import obj_jd_search


# Defining the blueprint 'auth'
mod_vector_search = Blueprint("vector_search", __name__, url_prefix='/vector_search')


@mod_vector_search.errorhandler(404)
def bad_request(error="Not Found"):
    return jsonify(error=error), 404


@mod_vector_search.route("/health", methods=['GET'])
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


@mod_vector_search.route("/insert_into_db", methods=['POST'])
def insert_jd_file_into_db() -> Response:
    """
    This method is used to parse the jd as a file and
    insert it into the vector db.
    @return: JSON
    """
    try:
        api_form_data = request.form
        content_type = api_form_data['contentType']
        org_code = api_form_data['orgCode']
        industry = api_form_data['industry']
        uuid = api_form_data.get('uuid')

        file_path = 'app/jd_vector_search/files/'
        if 'file' in request.files:
            txt_file = request.files['file']
            if txt_file.filename != '' and txt_file.filename[-3:] == 'txt':
                txt_file.save(file_path + txt_file.filename)
            else:
                return responseHandler.failure_response(
                    'TXT file not found in the form data.',
                    500
                )
        else:
            return responseHandler.failure_response(
                'TXT file not found in the form data.',
                500
            )

        # Extract text from .txt file
        jd_path = file_path + txt_file.filename
        jd_text = read_jd_file(jd_path)
        # Extract filename from filepath
        jd_name = jd_path.split('/')[-1]

        # Extract role from jd filename
        if '_' in jd_name:
            role = jd_name.split('_')[0]
            role = role.replace('Full-time', '')
        else:
            role = jd_name.split('.txt')[0].strip('_')

        # Process the JD
        jd_status = process_jd(
            jd_text=jd_text,
            role=role,
            content_type=content_type,
            org_code=org_code,
            industry=industry,
            uuid=uuid
        )

        # Delete the save JD file
        remove(file_path + txt_file.filename)

        if 'error' in jd_status:
            return responseHandler.failure_response(
                jd_status,
                400
            )
        else:
            success_response = {
                content_type: [
                    {
                        'documents': [
                            jd_status
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


@mod_vector_search.route("/insert_content_into_db", methods=['POST'])
def insert_jd_content_into_db() -> Response:
    """
    This method is used to parse the jd content as plain text/ html and
    insert it into the vector db.
    @return: JSON
    """
    try:
        api_request_data = json.loads(request.data)
        data = api_request_data['data']

        uuid = data['Job Description'].get('UUID')
        jd_text = data['Job Description'].get('innerText')
        role = data['Job Description'].get('role')
        content_type = data['contentType']
        org_code = data['orgCode']
        industry = data['industry']

        # Process the JD
        jd_status = process_jd(
            jd_text=jd_text,
            role=role,
            content_type=content_type,
            org_code=org_code,
            industry=industry,
            uuid=uuid
        )

        if 'error' in jd_status:
            return responseHandler.failure_response(
                jd_status,
                400
            )
        else:
            success_response = {
                content_type: [
                    {
                        'documents': [
                            jd_status
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


@mod_vector_search.route("/search", methods=['POST'])
def semantic_search() -> Response:
    """
    This method is used for vector search
    based on the given query.
    @return: JSON
    """
    try:

        api_request_data = json.loads(request.data)
        data = api_request_data['inputStream']

        org_code = api_request_data['orgCode']
        industry = api_request_data['industry']
        response = {}
        for attribute in data.keys():
            res = obj_jd_search.process_query(data[attribute]['topic'], data[attribute]['keywords'], org_code, industry)
            response[attribute] = res

        return responseHandler.success_response(
            response,
            200
        )
    except Exception as err:
        logger.error(err.args)
        return responseHandler.failure_response(
            ErrorMessages.SOMETHING_WENT_WRONG.value,
            500
        )
