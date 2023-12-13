import openai
from typing import List, Dict
from ast import literal_eval
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from app.utilities.logger import logger
from config import JD_METADATA_PROMPT, OPENAI_API_KEY, EMBEDDING_MODEL, JD_EXTRACTION_MODEL, QDRANT_URL, \
    QDRANT_API_KEY, QDRANT_COLLECTION_NAME, JD_FEATURES_TO_PREPARE_KEYWORDS

openai.api_key = OPENAI_API_KEY


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


def read_jd_file(file_path: str) -> str:
    """
    This function reads the text content
    from a .txt file
    @param file_path: str
    :return str
    """
    try:
        file_text = open(file_path, 'r').read()
        return file_text.strip()
    except Exception as err:
        logger.error('Error while reading the JD text file:', err.args)
        raise Exception(err)


def extract_metadata_from_jd(content: str) -> dict:
    """
    This function takes the jd text and calls the openAI
    GPT-3 to extract the metadata from it.
    @param content: str
    :return: dict
    """
    try:
        content = content.strip('\n').strip().strip('\n')
        response = openai.ChatCompletion.create(
            model=JD_EXTRACTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Job Description parser assistant."
                },
                {
                    "role": "user",
                    "content": JD_METADATA_PROMPT.replace('{{}}', '"""' + content + '"""')
                }
            ],
            temperature=0.4,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Convert response text to a valid json
        metadata_dict = str_to_dict(response['choices'][0]['message']['content'])
        return metadata_dict
    except Exception as err:
        logger.error('Error while extracting metadata from jd text using GPT-3:', err.args)
        raise Exception(err)


def prepare_data_for_vector_db(metadata: dict, content_type: str, org_code: str,
                               role: str, industry: str, uuid: str) -> dict:
    """
    This function transforms the extracted metadata to make
    it suitable to be inserted into the vector db.
    @param metadata: dictionary
    @param content_type: str
    @param org_code: str
    @param role: str
    @param industry: str
    @param uuid: str
    :return: dict
    """
    try:
        keywords = ''
        for key in metadata.keys():
            if key in JD_FEATURES_TO_PREPARE_KEYWORDS:
                keywords += '\n'.join(metadata[key]) + '\n'
        keywords = keywords.strip()

        if uuid is None:
            uuid = str(uuid4())

        data_dict = {
            'uuid': uuid,
            'contentType': content_type,
            'orgCode': org_code,
            'industry': industry,
            'topic': role.strip(),
            'keywords': keywords.strip(),
            'docMetadata': metadata
        }

        return data_dict

    except Exception as err:
        logger.error('Error while preparing the data for vector db:', err.args)
        raise Exception(err)


def encode_query(query: str) -> List:
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


def insert_vectors_into_db(data_dict: Dict) -> str:
    """
    This function is used to insert the data into the
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
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=data_dict['uuid'],
                    vector={
                        "topic": data_dict["topic_vector"],
                        "keywords": data_dict["keywords_vector"],
                    },
                    payload={
                        "contentType": data_dict['contentType'],
                        "orgCode": data_dict["orgCode"],
                        "industry": data_dict["industry"],
                        "topic": data_dict['topic'],
                        "keywords": data_dict['keywords'],
                        "docMetadata": data_dict["docMetadata"],
                    }
                )
            ]
        )
        jd_uuid = data_dict['uuid']
        return jd_uuid
    except Exception as err:
        logger.error('Error while inserting the data into the vector DB:', err.args)
        raise Exception(err)


def process_jd(jd_text: str, role: str, content_type: str, org_code: str, industry: str, uuid: str) -> dict:
    """
    This function process JD, calls different functions to extract information
    and load the vectors in to the vector database.
    @param jd_text: str
    @param role: str
    @param content_type: str
    @param org_code: str
    @param industry: str
    @param uuid: str
    :return: str
    """
    try:
        # Extract metadata from jd text
        metadata_dict = extract_metadata_from_jd(jd_text)
        if 'error' in metadata_dict:
            return metadata_dict
        # Prepare data to be inserted in vector DB
        data_dict = prepare_data_for_vector_db(metadata_dict, content_type, org_code, role, industry, uuid)

        # Convert the relevant columns into the vector
        data_dict['topic_vector'] = encode_query(data_dict['topic'])
        data_dict['keywords_vector'] = encode_query(data_dict['keywords'])

        # Insert the vectors into the vector db
        jd_uuid = insert_vectors_into_db(data_dict)
        return {
            'UUID': jd_uuid
        }
    except Exception as err:
        logger.error('Error while processing a jd:', err.args)
        raise Exception(err)

