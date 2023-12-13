"""Response handler for success and failure"""
import json
from flask import Response
from typing import Dict, AnyStr, Union


def success_response(response: Union[Dict, AnyStr], status: int) -> Response:
    """
    This method is used as a response
    handler for success responses.
    @param response: JSON/String
    @param status: Integer
    @return: JSON
    """
    data = {"data": response, "status": status}

    return Response(json.dumps(data), status=status,
                    mimetype='application/json')


def failure_response(error: Union[Dict, AnyStr], status: int) -> Response:
    """
    This method is used as a response
    handler for failure responses.
    @param error: JSON/String
    @param status: Integer
    @return: JSON
    """
    error = {"error": error, "status": status}
    return Response(json.dumps(error), status=status,
                    mimetype='application/json')
