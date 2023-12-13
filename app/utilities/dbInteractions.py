# This file is used to interact with database
import requests
import json

from app.utilities.logger import logger
from config import GET_ACTIVITY_TYPE_CONFIG, GET_CONTENT_TEMPLATE_CONFIG, GET_CAMPAIGN_CONSTRUCT_CONFIG


def get_activity_type_configuration(
        activity_type: str,
        implementation_key: str,
        org_id: str = None):
    """
    This method is used to fetch the
    configuration based on activity type passed.
    @param activity_type: str
    @param implementation_key: str
    @param org_id: str
    """
    try:
        headers = {
            'Content-Type': 'application/json'
        }

        payload = json.dumps({
            "data": {
                "org_id": org_id,
                "activity_type": activity_type,
                "implementation_key": implementation_key or "GPT3"
            }
        })

        response = requests.request(
            "POST",
            GET_ACTIVITY_TYPE_CONFIG,
            headers=headers,
            data=payload
        )

        return response.json()

    except Exception as err:
        logger.error(err.args)
        return {"error": err.args or "Something went wrong!", "status": 404}


def get_content_template(
        content_type: str,
        org_id: str = None):
    """
    This method is used to fetch the content template
    for the generative model based on content type passed.
    @param content_type: str
    @param org_id: str
    """
    try:
        headers = {
            'Content-Type': 'application/json'
        }

        payload = json.dumps({
            "data": {
                "org_id": org_id,
                "content_type": [content_type]
            }
        })

        response = requests.request(
            "POST",
            GET_CONTENT_TEMPLATE_CONFIG,
            headers=headers,
            data=payload
        )

        return response.json()

    except Exception as err:
        logger.error(err.args)
        return []


def get_campaign_construct(
        org_id: str = None):
    """
    This method is used to fetch the campaign construct
    to suggest suitable campaign based on dynamic events, such as news items.
    @param content_type: str
    @param org_id: str
    """
    try:
        headers = {
            'Content-Type': 'application/json'
        }

        payload = json.dumps({
            "data": {
                "org_id": org_id
            }
        })

        response = requests.request(
            "POST",
            GET_CAMPAIGN_CONSTRUCT_CONFIG,
            headers=headers,
            data=payload
        )

        return response.json()

    except Exception as err:
        logger.error(err.args)
        return []
