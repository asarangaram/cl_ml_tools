class VisualSearchEngineBase:
    def upsert(self, id: str, image_path: str):
        pass

    def search(self, image_path: str, top_k: int = 5):
        pass
