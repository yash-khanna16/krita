import openai
from typing import List
from qdrant_client.http import models
from ast import literal_eval
import numpy as np
from numpy import argmax
from qdrant_client import QdrantClient
from sklearn.metrics.pairwise import cosine_similarity

from app.utilities.logger import logger
from config import QDRANT_COLLECTION_NAME, OPENAI_API_KEY, EMBEDDING_MODEL, \
    QDRANT_URL, QDRANT_API_KEY, QDRANT_RESUME_COLLECTION_NAME, QDRANT_RESUME_MAX_SEARCH_LIMIT, \
    QDRANT_RESUME_MAX_OUTPUT_LIMIT, DEFAULT_RELEVANT_EXPERIENCE_RANGE, RESUME_RANKING_WEIGHTS, \
    MANDATORY_FEATURES_IN_JD, PREFERRED_FEATURES_IN_JD, \
    MOST_RELEVANT_FEATURES_IN_RESUME, LESS_RELEVANT_FEATURES_IN_RESUME

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
        return {'error': 'Cannot convert the resume metadata into a valid json',
                'content': content,
                'status': 400}


def fetch_jd_metadata_by_uuid(uuid: str) -> dict:
    """
    This function takes the uuid and searches for the JD
    in the vector database.
    @param uuid: str
    :return: dict
    """
    try:
        # Initializing the Qdrant client
        client = QdrantClient(
            url=QDRANT_URL,
            prefer_grpc=True,
            api_key=QDRANT_API_KEY,
        )
        # Fetching the JD by UUID
        data = client.retrieve(collection_name=QDRANT_COLLECTION_NAME, ids=[uuid])
        if len(data) == 0:
            return {'error': 'Given uuid does not exist in the vector db.'}
        jd_metadata = data[0].payload
        return jd_metadata
    except Exception as err:
        logger.error('Error while fetching the jd from the vector DB:', err.args)
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


def search_candidate_profiles_from_vector_db(
        mandatory_features_vectors: List,
        preferred_features_vectors: List,
        org_code: str,
        job_position: List,
        content_type: str,
        relevant_experience: List
) -> dict:
    """
    This function takes the encoded query and search the vector
    db for the most relevant result.
    @param mandatory_features_vectors: str,
    @param preferred_features_vectors: str,
    @param org_code: str,
    @param job_position: list
    @param content_type: str
    @param relevant_experience: list
    :return: dict
    """
    try:
        # Initializing the Qdrant client
        client = QdrantClient(
            url=QDRANT_URL,
            prefer_grpc=True,
            api_key=QDRANT_API_KEY,
        )
        if relevant_experience is None:
            relevant_experience = DEFAULT_RELEVANT_EXPERIENCE_RANGE

        results_limit = QDRANT_RESUME_MAX_SEARCH_LIMIT

        # Filter criteria
        filter_criteria = models.Filter(
            must=[
                models.FieldCondition(
                    key="orgCode",
                    match=models.MatchValue(value=org_code),
                ),
                models.FieldCondition(
                    key="jobPosition",
                    match=models.MatchAny(any=job_position),
                ),
            ]
        )

        # Search queries
        search_queries = [
            models.SearchRequest(
                vector={'name': 'most_relevant_features', 'vector': mandatory_features_vectors},
                filter=filter_criteria,
                limit=results_limit,
                with_vectors=False,
                with_payload=True
            ),
            models.SearchRequest(
                vector={'name': 'most_relevant_features', 'vector': preferred_features_vectors},
                filter=filter_criteria,
                limit=results_limit,
                with_vectors=False,
                with_payload=False
            ),
            models.SearchRequest(
                vector={'name': 'less_relevant_features', 'vector': mandatory_features_vectors},
                filter=filter_criteria,
                limit=results_limit,
                with_vectors=False,
                with_payload=False
            ),
            models.SearchRequest(
                vector={'name': 'less_relevant_features', 'vector': preferred_features_vectors},
                filter=filter_criteria,
                limit=results_limit,
                with_vectors=False,
                with_payload=False
            )
        ]

        # Perform the search from the vector DB
        vector_search_results = client.search_batch(
            collection_name=QDRANT_RESUME_COLLECTION_NAME,
            requests=search_queries
        )

        if len(vector_search_results[0]) == 0:
            return {
                content_type: [
                    {
                        "documents": []
                    }
                ]
            }

        # Process the search results
        output_document_list = process_vector_search_results(vector_search_results, results_limit,
                                                             mandatory_features_vectors, preferred_features_vectors)

        response = {
            content_type: [
                {
                    "documents": output_document_list[:QDRANT_RESUME_MAX_OUTPUT_LIMIT]
                }
            ]
        }

        return response
    except Exception as err:
        logger.error('Error while searching the candidate from vector db:', err.args)
        raise Exception(err)


def process_vector_search_results(vector_search_results: List, results_limit: int, mandatory_features_vectors: List,
                                  preferred_features_vectors: List):
    """
    This function processes the results from the vector search
    for multi vector ranking and constructing the output.
    @param vector_search_results: list
    @param results_limit: int
    @param mandatory_features_vectors: list
    @param preferred_features_vectors: list
    :return: list
    """
    most_relevant_results = vector_search_results[0]
    most_relevant_results_map = {}
    # less_relevant_results_map = {}
    # Instantiate a map keyed by the document id and value as array of scores
    vector_scores_map = {}

    if results_limit > len(most_relevant_results):
        results_limit = len(most_relevant_results)

    for row_index in range(results_limit):
        most_relevant_results_map[most_relevant_results[row_index].id] = {
            'score': most_relevant_results[row_index].score,
            'metadata': most_relevant_results[row_index].payload['metadata'],
            'relevant_features_vectors': most_relevant_results[row_index].payload['relevant_features_vectors']
        }

        # Set the map key as the document id and value as array of the score for each vector search
        result_vectors_count = len(vector_search_results)
        for vector_index in range(result_vectors_count):
            # Instantiate the array of vector scores if not already done
            document = vector_search_results[vector_index][row_index]
            if vector_scores_map.get(document.id) is None:
                vector_scores_map[document.id] = []
            # Set the vector score as per the vector index for the document id
            vector_scores_map[document.id].insert(vector_index, document.score)

    most_relevant_results_ids = list(most_relevant_results_map.keys())
    most_relevant_results_scores = [most_relevant_results_map[key]['score'] for key in most_relevant_results_map]

    modified_scores = []
    vector_weights = RESUME_RANKING_WEIGHTS

    for row_index in range(len(most_relevant_results_scores)):
        weighted_score = 0
        primary_document_id = most_relevant_results_ids[row_index]

        # Iterate through each vector (if exists) for the document
        for vector_index in range(result_vectors_count):
            # Check if the document was retrieved when querying on this vector
            if vector_index in range(len(vector_scores_map[primary_document_id])):
                weighted_score += vector_scores_map[primary_document_id][vector_index] * vector_weights[vector_index]
            else:
                # If the document was not retrieved, set the vector score for the document to 0
                vector_scores_map[primary_document_id].insert(vector_index, 0.0)

        modified_scores.append(weighted_score)

    # Combine scores and document IDs into a list of tuples
    score_document_pairs = list(zip(modified_scores, most_relevant_results_ids))

    # Sort the list of tuples in descending order based on scores
    sorted_pairs = sorted(score_document_pairs, reverse=True)

    # Create a dictionary with document IDs as keys and sorted scores as values
    modified_scores_sorted_dict = {doc_id: score for score, doc_id in sorted_pairs}

    output_document_list = []

    # Iterate through the primary vector results to construct the output
    for id in modified_scores_sorted_dict:

        # Generate feature wise similarity scores
        feature_match_score_map = {}
        resume_feature_vector_map = most_relevant_results_map[id]['relevant_features_vectors']
        for n_feature in resume_feature_vector_map:
            if len(resume_feature_vector_map[n_feature]) == 0:
                feature_match_score_map[n_feature] = -1
                continue
            # Calculating cosine similarity
            mandatory_feature_cosine = cosine_similarity(np.array(resume_feature_vector_map[n_feature]).reshape(1, -1),
                                                         np.array(mandatory_features_vectors).reshape(1, -1))[0][0]
            preferred_feature_cosine = cosine_similarity(np.array(resume_feature_vector_map[n_feature]).reshape(1, -1),
                                                         np.array(preferred_features_vectors).reshape(1, -1))[0][0]

            aggregated_score = (0.75 * mandatory_feature_cosine) + (0.25 * preferred_feature_cosine)

            feature_match_score_map[n_feature] = aggregated_score

        output_document_list.append(
            {
                "documentId": id,
                "documentRelevanceScore": modified_scores_sorted_dict[id],
                "vectorMatchScore": vector_scores_map[id],
                "totalRelevantExperience": most_relevant_results_map[id].get('total_relevant_experience'),
                "metadata": most_relevant_results_map[id]['metadata'],
                "featureMatchScore": feature_match_score_map
            }
        )

    return output_document_list


def process_search_results(results_limit, most_relevant_results, less_relevant_results):
    """
    This function processes the results from the vector search
    for ranking and constructing the output.
    @param results_limit: int,
    @param most_relevant_results: str,
    @param less_relevant_results: str,
    :return: list
    """
    most_relevant_results_map = {}
    less_relevant_results_map = {}

    if results_limit > len(most_relevant_results):
        results_limit = len(most_relevant_results)

    for i in range(results_limit):
        most_relevant_results_map[most_relevant_results[i].id] = {
            'score': most_relevant_results[i].score,
            'metadata': most_relevant_results[i].payload['metadata']
        }

        less_relevant_results_map[less_relevant_results[i].id] = less_relevant_results[i].score

    most_relevant_results_ids = list(most_relevant_results_map.keys())
    most_relevant_results_scores = [most_relevant_results_map[key]['score'] for key in most_relevant_results_map]

    less_relevant_results_ids = list(less_relevant_results_map.keys())
    less_relevant_results_scores = list(less_relevant_results_map.values())

    modified_scores = []

    for i in range(len(most_relevant_results_scores)):
        if most_relevant_results_ids[i] in less_relevant_results_ids:
            keyword_index = less_relevant_results_ids.index(most_relevant_results_ids[i])
            weighted_score = (most_relevant_results_scores[i] * 0.75) + (
                    less_relevant_results_scores[keyword_index] * 0.25)
        else:
            weighted_score = most_relevant_results_scores[i] * 0.75

        modified_scores.append(weighted_score)

        # Combine scores and document IDs into a list of tuples
    score_document_pairs = list(zip(modified_scores, most_relevant_results_ids))

    # Sort the list of tuples in descending order based on scores
    sorted_pairs = sorted(score_document_pairs, reverse=True)

    # Create a dictionary with document IDs as keys and sorted scores as values
    modified_scores_sorted_dict = {doc_id: score for score, doc_id in sorted_pairs}

    output_document_list = []

    for ids in modified_scores_sorted_dict:
        vector_match_score = [most_relevant_results_map[ids]['score']]
        if less_relevant_results_map.get(ids):
            vector_match_score.append(less_relevant_results_map[ids])
        else:
            vector_match_score.append(0.0)

        output_document_list.append(
            {
                "documentId": ids,
                "documentRelevanceScore": modified_scores_sorted_dict[ids],
                "vectorMatchScore": vector_match_score,
                "totalRelevantExperience": most_relevant_results_map[ids].get('total_relevant_experience'),
                "metadata": most_relevant_results_map[ids]['metadata']
            }
        )

    return output_document_list


def search_profiles(doc_uuid: str, content_type: str, org_code: str, job_position: List,
                    search_keywords: List = [], relevant_experience: List = None,
                    reference_profiles: List = []) -> dict:
    """
    This function takes the job uuid and calls relevant functions like
    fetching JD metadata, converting into vectors and search relevant resume.
    @param doc_uuid: str
    @param content_type: str
    @param org_code: str
    @param job_position: list
    @param search_keywords: list
    @param relevant_experience: list
    @param reference_profiles: list
    :return: str
    """
    try:
        # Fetch JD metadata using job uuid
        jd_metadata = fetch_jd_metadata_by_uuid(doc_uuid)

        if 'error' in jd_metadata:
            return jd_metadata

        # Preparing data for matching the profile
        mandatory_features, preferred_features = get_features_to_match(jd_metadata, search_keywords, reference_profiles)

        # Encode the queries
        mandatory_features_vectors = encode_query(mandatory_features)
        preferred_features_vectors = encode_query(preferred_features)

        # Search the resume vector DB for finding the candidate profile
        response = search_candidate_profiles_from_vector_db(mandatory_features_vectors, preferred_features_vectors,
                                                            org_code, job_position, content_type, relevant_experience)

        return response
    except Exception as err:
        logger.error('Error while searching candidate profiles:', err.args)
        raise Exception(err)


def get_features_to_match(jd_metadata: dict, search_keywords: List, reference_profiles: List):
    """
    This function takes the JD and reference profiles, and
    gets the mandatory and preferred features to match.
    @param jd_metadata: dict
    @param search_keywords: list
    @param reference_profiles: list
    :return: str
    """
    try:
        mandatory_features = ''
        preferred_features = ''

        for m_jd_feature in MANDATORY_FEATURES_IN_JD:
            mandatory_features += '\n'.join(jd_metadata['docMetadata'][m_jd_feature]).strip()
        for p_jd_feature in PREFERRED_FEATURES_IN_JD:
            preferred_features = '\n'.join(jd_metadata['docMetadata'][p_jd_feature]).strip()

        # Fetch reference profiles for including in the features to match
        if reference_profiles is not None and len(reference_profiles) > 0:
            mandatory_features += '\n'
            preferred_features += '\n'

            for resume_uuid in reference_profiles:
                # Fetch reference profiles metadata using the resume uuid
                resume_metadata = fetch_resume_metadata_by_uuid(resume_uuid)

                for m_resume_feature in MOST_RELEVANT_FEATURES_IN_RESUME:
                    mandatory_features += '\n'.join(resume_metadata['metadata'][m_resume_feature]).strip()
                for l_resume_feature in LESS_RELEVANT_FEATURES_IN_RESUME:
                    preferred_features += '\n'.join(resume_metadata['metadata'][l_resume_feature]).strip()

        if search_keywords is not None and len(search_keywords) > 0:
            mandatory_features += '\n' + '\n'.join(search_keywords).strip()
            preferred_features += '\n' + '\n'.join(search_keywords).strip()

        return mandatory_features, preferred_features
    except Exception as err:
        logger.error('Error while searching candidate profiles:', err.args)
        raise Exception(err)


def fetch_resume_metadata_by_uuid(uuid: str) -> dict:
    """
    This function takes the uuid and searches for the resume
    in the vector database.
    @param uuid: str
    :return: dict
    """
    try:
        # Initializing the Qdrant client
        client = QdrantClient(
            url=QDRANT_URL,
            prefer_grpc=True,
            api_key=QDRANT_API_KEY,
        )
        # Fetching the candidate profile by UUID
        data = client.retrieve(collection_name=QDRANT_RESUME_COLLECTION_NAME, ids=[uuid])
        if len(data) == 0:
            return {'error': 'Given uuid does not exist in the vector db.'}
        resume_metadata = data[0].payload
        return resume_metadata
    except Exception as err:
        logger.error('Error while fetching the resume from the vector DB:', err.args)
        raise Exception(err)
