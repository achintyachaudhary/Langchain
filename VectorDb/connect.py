from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document
from dotenv import load_dotenv
import os
load_dotenv()

QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
class QdrantConnect:
    def __init__(self):
        # connect to Qdrant Cloud
        self.client = QdrantClient(
            url=QDRANT_CLUSTER_URL,
            api_key=QDRANT_API_KEY,
            cloud_inference=True
        )

    def query_db(self, collection, query,  model="sentence-transformers/all-MiniLM-L6-v2" ):

        # search for similar menu items
        results = self.client.query_points(
            collection_name=collection,
            query=Document(text=query, model=model),
            with_payload=True,
            limit=1
        )

        return results.points[0]

    def save_db(self, collection, documents):
        # upsert points to collection
        # points generator
        points = []
        for i, menu_item in enumerate(documents):
            point = PointStruct(
                id=i,
                vector=Document(
                    text=menu_item,
                    model="sentence-transformers/all-MiniLM-L6-v2"
                ),
                payload={
                    "text": menu_item,
                }
            )
            points.append(point)

        self.client.upsert(
        collection_name=collection,
        points=points,
        )