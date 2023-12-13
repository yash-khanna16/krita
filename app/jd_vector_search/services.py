from qdrant_client import QdrantClient
from qdrant_client.http import models
from numpy import argmax
import openai
from os import getenv
from typing import List
from app.utilities.logger import logger

from config import QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL, QDRANT_COLLECTION_NAME

# Setting openai API key
openai.api_key = getenv("OPENAI_API_KEY")


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


class JdVectorSearch:
    """
    Class used for jd vector search based on the given query.
    """

    def __init__(self):
        """
        Loading the embedding model and initializing the Qdrant client.
        """
        try:
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
        except Exception as err:
            logger.error('Error initializing the Qdrant client:', str(err))
            # raise Exception(err)

    def process_query(self, topic_query: str, keywords_query: List,
                      org_code: str, industry: str) -> List:
        """
        This method is used to perform the vector search
        based on the given query.
        :param topic_query: str
        :param keywords_query: List of keywords
        :param org_code: str
        :param industry: str
        :return: List
        """
        try:
            # prepare queries
            keywords_query = '\n'.join(keywords_query)

            # Encode the queries to vectors
            topic_query_vector = encode_query(topic_query)
            keywords_query_vector = encode_query(keywords_query)

            results_limit = 3

            # Perform a vector search from a vector DB
            topic_results = self.client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=('topic', topic_query_vector),
                query_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="orgCode",
                            match=models.MatchValue(value=org_code),
                        ),
                        models.FieldCondition(
                            key="industry",
                            match=models.MatchValue(value=industry),
                        ),
                    ]
                ),
                limit=results_limit,
                with_vectors=False,
                with_payload=True,
            )

            keywords_results = self.client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=('keywords', keywords_query_vector),
                query_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="orgCode",
                            match=models.MatchValue(value=org_code),
                        ),
                        models.FieldCondition(
                            key="industry",
                            match=models.MatchValue(value=industry),
                        ),
                    ]
                ),
                limit=results_limit,
                with_vectors=False,
                with_payload=True,
            )

            if len(topic_results) == 0 or len(keywords_results) == 0:
                return [
                    {
                        "documents": []
                    }
                ]

            topic_map = {}
            keywords_map = {}

            for i in range(results_limit):
                topic_map[topic_results[i].id] = {
                    'score': topic_results[i].score,
                    'industry': topic_results[i].payload['industry'],
                    'orgCode': topic_results[i].payload['orgCode'],
                    'role': topic_results[i].payload['topic'],
                    'docMetadata': topic_results[i].payload['docMetadata']
                }

                keywords_map[keywords_results[i].id] = keywords_results[i].score

            topic_ids = list(topic_map.keys())
            topic_scores = [topic_map[key]['score'] for key in topic_map]

            keywords_ids = list(keywords_map.keys())
            keywords_scores = list(keywords_map.values())

            modified_scores = []

            for i in range(len(topic_scores)):
                if topic_ids[i] in keywords_ids:
                    keyword_index = keywords_ids.index(topic_ids[i])
                    weighted_score = (topic_scores[i] * 0.75) + (keywords_scores[keyword_index] * 0.25)
                else:
                    weighted_score = topic_scores[i] * 0.75

                modified_scores.append(weighted_score)

            # getting maximum modified score
            max_arg_modified_score = argmax(modified_scores)

            # document id with max score
            max_score_doc_id = topic_ids[max_arg_modified_score]

            vector_match_score = [topic_map[max_score_doc_id]['score']]
            if keywords_map.get(max_score_doc_id):
                vector_match_score.append(keywords_map[max_score_doc_id])
            else:
                vector_match_score.append(0.0)

            doc_metadata_sections = []
            for key in eval(str(topic_map[max_score_doc_id]['docMetadata'])):
                doc_metadata_sections.append(
                    {
                        'sectionId': key,
                        'text': '\n'.join(eval(str(topic_map[max_score_doc_id]['docMetadata']))[key]).strip('\n').strip()
                    }
                )

            response = [
                {
                    "documents": [
                        {
                            "documentId": max_score_doc_id,
                            "documentRelevanceScore": modified_scores[max_arg_modified_score],
                            "vectorMatchScore": vector_match_score,
                            "docMetadata": {
                                "industry": topic_map[max_score_doc_id]['industry'],
                                "org_code": topic_map[max_score_doc_id]['orgCode'],
                                "role": topic_map[max_score_doc_id]['role'],
                                "sections": doc_metadata_sections
                            }
                        }
                    ]
                }
            ]

            return response

        except Exception as err:
            logger.error('Error while processing the search query:', str(err))
            raise Exception(err)
