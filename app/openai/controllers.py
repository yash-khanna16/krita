import json

from flask import (
    Blueprint, jsonify, Response, request
)

from app.utilities import responseHandler
from app.utilities.constants import ErrorMessages, SuccessMessages
from app.utilities.logger import logger
from app.openai.services import OpenAIService


# Defining the blueprint 'auth'
mod_openai = Blueprint("openai", __name__, url_prefix='/openai')


@mod_openai.errorhandler(404)
def bad_request(error="Not Found"):
    return jsonify(error=error), 404


@mod_openai.route("/health", methods=['GET'])
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


@mod_openai.route("/generate", methods=['POST'])
def generate_content() -> Response:
    """
    This method is used for generating
    content for user using OPENAI GPT-3 API.
    @return: JSON
    """
    try:

        api_request_data = json.loads(request.data)
        res = OpenAIService.generate_content_service(api_request_data.get('data'), api_request_data.get('user'))
        return responseHandler.success_response(
            res,
            200
        )
    except Exception as err:
        logger.error(err.args)
        return responseHandler.failure_response(
            ErrorMessages.SOMETHING_WENT_WRONG.value,
            500
        )
