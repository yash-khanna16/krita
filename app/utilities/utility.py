import re
from ast import literal_eval
from app.utilities.logger import logger


def get_keys_from_prompt(prompt):
    return re.findall(r"{(\w+)}", prompt)


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
        return {'error': 'Cannot convert the content metadata into a valid json',
                'content': content,
                'status': 400}


def get_int_from_var(elapsed_time_obj) -> int:
    """
    This function is used to get an integer from an input variable
    that could be a string or integer.
    :param query: str
    :return: int
    """
    if isinstance(elapsed_time_obj, str) and elapsed_time_obj.isdigit():
        elapsed_time = literal_eval(elapsed_time_obj)
    elif isinstance(elapsed_time_obj, int):
        elapsed_time = elapsed_time_obj
    else:
        elapsed_time = 0
    return elapsed_time


def transform_keyvalue_list_to_string(data: list) -> str:
    """
    This function is used to transform a JSON list of label-value pairs
    to a string representation.
    :param query: list
    :return: str
    """
    output = ""
    for item in data:
        output += "{label}: {value}\r\n".format(label=item['label'], value=item['value'])
    return output.rstrip()
