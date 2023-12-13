import openai
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from typing import List, Dict
from io import StringIO
from ast import literal_eval
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from app.utilities.logger import logger
from app.utilities.utility import get_int_from_var, str_to_dict
from config import RESUME_METADATA_PROMPT, RELEVANT_EXPERIENCE_THRESHOLD, MOST_RECENT_EXPERIENCE_THRESHOLD, \
    OPENAI_API_KEY, EMBEDDING_MODEL, RESUME_EXTRACTION_MODEL, RESUME_EXTRACTION_MODEL_ALT, RESUME_EXTRACTION_TOKEN_THRESHOLD, \
    QDRANT_URL, QDRANT_API_KEY, QDRANT_RESUME_COLLECTION_NAME,\
    MOST_RELEVANT_FEATURES_IN_RESUME, LESS_RELEVANT_FEATURES_IN_RESUME

openai.api_key = OPENAI_API_KEY


def convert_pdf_to_txt(path: str) -> str:
    """
    This function takes the path of a pdf file
    and extract the text out of it.
    @param path: str
    :return: str
    """
    try:
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos=set()

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()

        fp.close()
        device.close()
        retstr.close()
        return text
    except Exception as err:
        logger.error('Error while extracting the text from a pdf file:', str(err))
        raise Exception(err)


def extract_metadata_from_resume(content: str) -> dict:
    """
    This function takes the resume text and calls the openAI
    GPT-3 to extract the metadata from it.
    @param content: str
    :return: dict
    """
    try:
        # Determine whether to use the primary or backup model based on content length (~ 4 char = 1 token)
        if len(content) < RESUME_EXTRACTION_TOKEN_THRESHOLD * 4:
            model_name = RESUME_EXTRACTION_MODEL
        else:
            model_name = RESUME_EXTRACTION_MODEL_ALT

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert resume parser assistant."
                },
                {
                    "role": "user",
                    "content": RESUME_METADATA_PROMPT.replace('{{}}', '"""' + content + '"""')
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
        logger.error('Error while extracting metadata from resume text using GPT-3:', err.args)
        raise Exception(err)


def prepare_data_for_vector_db(metadata: dict, content_type: str, org_code: str, job_position: List) -> dict:
    """
    This function transforms the extracted metadata to make
    it suitable to be inserted into the vector db.
    @param metadata: dictionary
    @param content_type: str
    @param org_code: str
    @param job_position: List
    :return: dict
    """
    try:
        # Get the project experience sections
        project_experiences = metadata.get('projectExperiences')

        total_experience = 0
        total_relevant_experience = 0
        # Calculate the total and relevant months of experience
        for obj in project_experiences:
            # Get the elapsed time
            elapsed_time_obj = obj.get("elapsedTime")
            elapsed_time = get_int_from_var(elapsed_time_obj)

            # Include elapsed time in the total experience
            total_experience += elapsed_time
            # Add the elapsed time in relevant experience only if at least one relevant skill found in the project experience
            most_relevant_skills = obj.get("mostRelevantTechnicalAndOperationalSkills")
            if isinstance(most_relevant_skills, list) and len(most_relevant_skills) > 0:
                total_relevant_experience += elapsed_time

        current_relevant_experience = 0
        most_relevant_skills_and_roles = ''
        less_relevant_skills_and_roles = ''
        most_relevant_feature_list = MOST_RELEVANT_FEATURES_IN_RESUME
        less_relevant_feature_list = LESS_RELEVANT_FEATURES_IN_RESUME

        # If the total experience is less than the recency threshold or relevant experience is more than a certain
        # threshold, include the most relevant experiences directly
        if total_experience < MOST_RECENT_EXPERIENCE_THRESHOLD or total_relevant_experience > (total_experience * RELEVANT_EXPERIENCE_THRESHOLD):
            for feature in most_relevant_feature_list:
                most_relevant_skills_and_roles += '\n'.join(metadata[feature]) + '\n'
        else:
            for experience_item in project_experiences:
                current_relevant_experience += int(experience_item.get('elapsedTime') or "0")
                # If the relevant experience is less than a threshold of the total experience, include the most
                # relevant experiences only for the latest projects (within threshold)
                if current_relevant_experience < (total_experience * RELEVANT_EXPERIENCE_THRESHOLD):
                    most_relevant_skills_and_roles += metadata['mostRelevantTechnicalAndOperationalSkills'] + '\n'
            # # Add the most relevant roles
            # most_relevant_skills_and_roles += metadata['mostRelevantRoles'] + '\n'

        # Add the less relevant experiences
        for feature in less_relevant_feature_list:
            less_relevant_skills_and_roles += '\n'.join(metadata[feature]) + '\n'

        # Include the experience in metadata and also add as separate columns for filtering
        metadata['totalExperience'] = total_experience
        metadata['totalRelevantExperience'] = total_relevant_experience or 0

        data_dict = {
            'uuid': str(uuid4()),
            'contentType': content_type,
            'orgCode': org_code,
            'jobPosition': job_position,
            'most_relevant_features': most_relevant_skills_and_roles.strip(),
            'less_relevant_features': less_relevant_skills_and_roles.strip(),
            'total_experience': total_experience,
            'total_relevant_experience': total_relevant_experience or 0,
            'metadata': metadata
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


def insert_vectors_into_db(data_dict: Dict, feature_vector_map: Dict) -> str:
    """
    This function is used to insert the data into the
    vector db using Qdrant.
    @param data_dict: dict
    @param feature_vector_map: dict
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
            collection_name=QDRANT_RESUME_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=data_dict['uuid'],
                    vector={
                        "most_relevant_features": data_dict["most_relevant_features_vectors"],
                        "less_relevant_features": data_dict["less_relevant_features_vectors"],
                    },
                    payload={
                        "contentType": data_dict['contentType'],
                        "orgCode": data_dict["orgCode"],
                        "jobPosition": data_dict["jobPosition"],
                        "most_relevant_features": data_dict['most_relevant_features'],
                        "less_relevant_features": data_dict['less_relevant_features'],
                        "metadata": data_dict["metadata"],
                        "relevant_features_vectors": feature_vector_map
                    }
                )
            ]
        )
        resume_uuid = data_dict['uuid']
        return resume_uuid
    except Exception as err:
        logger.error('Error while inserting the data into the vector DB:', err.args)
        raise Exception(err)


def process_resume(resume_path: str, content_type: str, org_code: str, job_position: List) -> dict:
    """
    This function process resume, calls different functions to extract information
    and load the vectors in to the vector database.
    @param resume_path: str,
    @param content_type: str,
    @param org_code: str,
    @param job_position: list
    :return: str
    """
    try:
        # Extract text from pdf file
        resume_text = convert_pdf_to_txt(resume_path)
        # Extract metadata from resume text
        metadata_dict = extract_metadata_from_resume(resume_text)
        if 'error' in metadata_dict:
            return metadata_dict
        # Prepare data to be inserted in vector DB
        data_dict = prepare_data_for_vector_db(metadata_dict, content_type, org_code, job_position)

        # Convert the relevant columns into the vector
        data_dict['most_relevant_features_vectors'] = encode_query(data_dict['most_relevant_features'])
        data_dict['less_relevant_features_vectors'] = encode_query(data_dict['less_relevant_features'])

        # Generating the feature map for most relevant and less relevant features extracted
        feature_vector_map = {}
        for n_feature in MOST_RELEVANT_FEATURES_IN_RESUME:
            if len(metadata_dict[n_feature]) == 0:
                feature_vector_map[n_feature] = []
            else:
                feature_vector_map[n_feature] = encode_query('\n'.join(metadata_dict[n_feature]).strip())
        for n_feature in LESS_RELEVANT_FEATURES_IN_RESUME:
            if len(metadata_dict[n_feature]) == 0:
                feature_vector_map[n_feature] = []
            else:
                feature_vector_map[n_feature] = encode_query('\n'.join(metadata_dict[n_feature]).strip())

        # Insert the vectors into the vector db
        resume_uuid = insert_vectors_into_db(data_dict, feature_vector_map)
        return {
            'UUID': resume_uuid,
            'profile': metadata_dict.get('profile'),
            'contact': metadata_dict.get('contact')
        }
    except Exception as err:
        logger.error('Error while processing a resume:', err.args)
        raise Exception(err)

