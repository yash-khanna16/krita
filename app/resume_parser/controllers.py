from flask import (
    Blueprint, jsonify, Response, request
)
import json
from os import remove
from app.utilities import responseHandler
from app.utilities.constants import ErrorMessages, SuccessMessages
from app.utilities.logger import logger

from app.resume_parser.resume_vectordb_insertion import process_resume
from app.resume_parser.search_candidate_profiles import search_profiles

from config import DEFAULT_RELEVANT_EXPERIENCE_RANGE


# Defining the blueprint 'auth'
mod_resume_parser = Blueprint("resume_parser", __name__, url_prefix='/resume')


@mod_resume_parser.errorhandler(404)
def bad_request(error="Not Found"):
    return jsonify(error=error), 404


@mod_resume_parser.route("/health", methods=['GET'])
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


@mod_resume_parser.route("/insert_into_db", methods=['POST'])
def insert_resume_data_into_db() -> Response:
    """
    This method is used for vector search
    based on the given query.
    @return: JSON
    """
    try:
        api_form_data = request.form
        content_type = api_form_data['contentType']
        org_code = api_form_data['orgCode']
        job_position = api_form_data['jobPosition']

        file_path = 'app/resume_parser/files/'
        if 'file' in request.files:
            pdf_file = request.files['file']
            if pdf_file.filename != '' and pdf_file.filename[-3:] == 'pdf':
                pdf_file.save(file_path + pdf_file.filename)
            else:
                return responseHandler.failure_response(
                    'PDF file not found in the form data.',
                    500
                )
        else:
            return responseHandler.failure_response(
                'PDF file not found in the form data.',
                500
            )

        # Process the resume
        resume_status = process_resume(
            resume_path=file_path+pdf_file.filename,
            content_type=content_type,
            org_code=org_code,
            job_position=eval(job_position)
        )

        # Delete the save resume file
        remove(file_path + pdf_file.filename)

        if 'error' in resume_status:
            return responseHandler.failure_response(
                resume_status,
                400
            )
        else:
            success_response = {
                content_type: [
                    {
                        'documents': [
                            resume_status
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


@mod_resume_parser.route("/search_profiles", methods=['POST'])
def search_candidate_profiles() -> Response:
    """
    This method is used for vector search
    based on the given query.
    @return: JSON
    """
    try:
        api_request_data = json.loads(request.data)
        data = api_request_data['data']
        doc_uuid = data['Job Description']['docUID']

        search_keywords = []
        relevant_experience = DEFAULT_RELEVANT_EXPERIENCE_RANGE
        reference_profiles = None
        if data['Job Description'].get('criteria') is not None:
            search_keywords = data['Job Description']['criteria'].get('keywords')
            relevant_experience = data['Job Description']['criteria'].get('relevantExperience')
            reference_profiles = data['Job Description']['criteria'].get('referenceProfiles')
        
        content_type = api_request_data['contentType']
        org_code = api_request_data['orgCode']
        job_position = api_request_data['jobPosition']

        most_relevant_match = search_profiles(doc_uuid, content_type, org_code, job_position, search_keywords,
                                              relevant_experience, reference_profiles)

        if 'error' in most_relevant_match:
            return responseHandler.failure_response(
                most_relevant_match['error'],
                500
            )

        return responseHandler.success_response(
            most_relevant_match,
            200
        )

    except Exception as err:
        logger.error('Error while searching candidate profiles:', err.args)
        return responseHandler.failure_response(
            ErrorMessages.SOMETHING_WENT_WRONG.value + 'Error while searching candidate profiles' + str(err.args),
            500
        )
