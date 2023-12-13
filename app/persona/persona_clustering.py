import openai
from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from app.utilities.logger import logger

from config import OPENAI_API_KEY, EMBEDDING_MODEL

openai.api_key = OPENAI_API_KEY


def encode_content_to_embeddings(text: str) -> List:
    """
    This function is used to encode text into
    vectors using openai model.
    :param text: str
    :return: List
    """
    try:
        # Convert the text to embeddings
        response = openai.Embedding.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        embeddings = response['data'][0]['embedding']
        return embeddings
    except Exception as err:
        logger.error('Error while converting text to embeddings:', str(err))
        raise Exception(err)


def handle_clustering(content_list: List) -> Tuple:
    """
    This function is used to preprocess the data,
    converting to embeddings and performing clustering
    :param content_list: List
    :return: Tuple
    """
    try:
        # converting each text content to embeddings
        embeddings = [encode_content_to_embeddings(str(data_point)) for data_point in content_list]

        max_clusters = 20 if len(content_list) > 20 else len(content_list)
        silhouette_avg = {}

        # generate pairwise cosine distances
        distances = pairwise_distances(embeddings, metric='cosine')

        max_silhouette_score = -99
        max_score_cluster_labels = []

        # Check for best silhouette Score
        for n_cluster in range(2, max_clusters):
            # initialise kmeans
            kmeans = KMeans(n_clusters=n_cluster, n_init='auto', random_state=104)
            kmeans.fit(distances)
            cluster_labels = kmeans.labels_

            # silhouette score
            score = silhouette_score(distances, cluster_labels, metric='precomputed')
            silhouette_avg[n_cluster] = score

            if score > max_silhouette_score:
                max_silhouette_score = score
                max_score_cluster = n_cluster
                max_score_cluster_labels = cluster_labels
            else:
                continue

        clusters_map = {}
        for data_num, cluster_num in enumerate(max_score_cluster_labels.tolist()):
            if cluster_num not in clusters_map:
                clusters_map[int(cluster_num)] = [content_list[data_num]]
            else:
                clusters_map[int(cluster_num)].append(content_list[data_num])

        # return max_silhouette_score, max_score_cluster_labels.tolist()
        return clusters_map, max_silhouette_score

    except Exception as err:
        logger.error('Error while clustering content:', err.args)
        raise Exception(err)





