from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class TradingKnowledgeBase:
    def __init__(self, data_folder="data", index_path="faiss_index.bin"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.data_folder = data_folder
        self.index_path = index_path
        self.documents = []
        self.index = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists("docs.txt"):
            self.index = faiss.read_index(self.index_path)
            with open("docs.txt", "r", encoding="utf-8") as f:
                self.documents = f.readlines()
        else:
            self.create_index()

    def create_index(self):
        texts = []
        for file in os.listdir(self.data_folder):
            path = os.path.join(self.data_folder, file)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read().strip())

        self.documents = texts
        embeddings = self.model.encode(texts)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        faiss.write_index(index, self.index_path)

        with open("docs.txt", "w", encoding="utf-8") as f:
            f.writelines([t + "\n" for t in texts])

        self.index = index

    def retrieve_context(self, query, top_k=2):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = [self.documents[i].strip() for i in indices[0]]
        return "\n\n".join(results)
