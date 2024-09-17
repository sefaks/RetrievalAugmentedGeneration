import unittest
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedings import get_embeddings

CHROMA_PATH = "data/chroma"

class TestChromaDatabase(unittest.TestCase):
    
    def setUp(self):
        """Testler başlamadan önce çalışacak kod."""
        self.embedding_function = get_embeddings()
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)

    def test_embeddings_function(self):
        """Embedding fonksiyonunu test et."""
        test_text = "Doğal unsurlara örnekler"
        embedding = self.embedding_function.embed_documents([test_text])[0]  # veya uygun olanı kullanın
        self.assertIsNotNone(embedding, "Embedding fonksiyonu None döndürdü")
        self.assertTrue(len(embedding) > 0, "Embedding uzunluğu 0'dan büyük olmalı")

    def test_add_documents_to_db(self):
        """Veritabanına belgeleri eklemeyi test et."""
        test_document = Document(
            page_content="Doğa ve insan etkileşimi hakkında bilgi.",
            metadata={"source": "test_source.pdf", "page": 1}
        )
        self.db.add_documents([test_document], ids=["test_id_1"])
        # Veritabanındaki belgeleri kontrol edin.
        documents = self.db.get(include=["documents", "metadatas"])
        # Belge ve metadata verisini alıp kontrol edin
        docs = documents.get("documents", [])
        self.assertTrue(any(doc.metadata.get("source") == "test_source.pdf" for doc in docs), "Test belgesi veritabanında bulunamadı")

    def test_query_rag(self):
        """Sorgu işlevini test et."""
        # Test için uygun bir belge ekleyin
        test_document = Document(
            page_content="Doğal unsurlara örnekler: toprak, su, bitki.",
            metadata={"source": "test_source.pdf", "page": 1}
        )
        self.db.add_documents([test_document], ids=["test_id_2"])
        # Sorgu yapın ve sonucu kontrol edin
        query_text = "Doğal unsurlara örnekler nelerdir?"
        results = self.db.similarity_search_with_score(query_text, k=5)
        self.assertGreater(len(results), 0, "Sorgu sonuçları beklenenden az veya boş")

if __name__ == "__main__":
    unittest.main()
