import faiss
import numpy as np
import datetime

class FAISSMemory:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.embeddings = np.empty((0, dim), dtype=np.float32)
        self.metadata = []

    def store_embedding(self, user_id, message, embedding):
        """Stores a user message as an embedding for context retrieval."""
        self.embeddings = np.vstack((self.embeddings, embedding))
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # Save metadata (who said it and when)
        self.metadata.append({"user_id": user_id, "message": message, "timestamp": datetime.datetime.now().isoformat()})

    def retrieve_similar_messages(self, query_embedding, user_id, top_k=3):
        """Finds past conversations that are similar to the current input."""
        if self.embeddings.shape[0] == 0:
            return []

        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        
        # ✅ Check for missing metadata before accessing
        return [
            self.metadata[i] for j, i in enumerate(indices[0])
            if i != -1 and i < len(self.metadata) and self.metadata[i]["user_id"] == user_id
        ]
