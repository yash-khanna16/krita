import json

from flask import (
    Blueprint, jsonify, Response, request
)

from app.utilities import responseHandler
from app.utilities.constants import ErrorMessages, SuccessMessages
from app.utilities.logger import logger
from app.persona.persona_clustering import handle_clustering

# Defining the blueprint 'auth'
mod_persona = Blueprint("persona", __name__, url_prefix='/persona')


@mod_persona.errorhandler(404)
def bad_request(error="Not Found"):
    return jsonify(error=error), 404


@mod_persona.route("/health", methods=['GET'])
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


@mod_persona.route("/generate_clusters", methods=['POST'])
def generate_clusters_of_data_points() -> Response:
    """
    This method is used to parse the input data
    and returns the generated clusters..
    @return: JSON
    """
    try:
        api_data = request.json
        content_list = list(api_data['content'])

        clusters_map, max_silhouette_score = handle_clustering(content_list)

        response = {
            "cluster_score": max_silhouette_score,
            "cluster_data": clusters_map
            }

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