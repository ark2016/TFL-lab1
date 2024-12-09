import yaml
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

import app.core.vector_db.text_translator as text_translator
import app.config.config as config
import app.schemas.questions as schemas

faiss_db = None


class FaissDB:
    model = None
    index = None

    def __init__(self, sentence_transformer_name: str):
        self.model = SentenceTransformer(sentence_transformer_name)

        # база знаний
        with open('data.yaml', 'r') as file:
            self.data = yaml.safe_load(file)

        self.index = faiss.read_index('vectorized_data.faiss')

    def search_similar(self, query, k_max=2, similarity_threshold=0.69):
        """
        Dynamic search for similar objects based on similarity threshold.
        :param query: query string
        :param k_max: maximum number of results to return
        :param similarity_threshold: threshold for similarity to dynamically adjust k
        :return: list of similar objects
        """
        translator = text_translator.translator

        query_embedding = self.model.encode([translator.translate_text(query)])

        faiss.normalize_L2(query_embedding)

        D, I = self.index.search(np.array(query_embedding), k_max)

        dynamic_k = 1
        for i in range(1, k_max):
            if D[0][i] < similarity_threshold:
                break
            dynamic_k += 1

        return [self.data[idx] for idx in I[0][:dynamic_k]]


def init_faiss_db():
    global faiss_db
    faiss_db = FaissDB(
        config.SingletonConfig.get_instance().get_sentence_transformer_name(),
    )


def make_question_answer(qa_item: dict) -> schemas.QuestionAnswer:
    return schemas.QuestionAnswer(question=qa_item["question"], answer=qa_item["answer"])


def process_question(question: str) -> list[schemas.QuestionAnswer]:
    """
    Process the question, finding similar knowledge base elements
    :param question:
    :return: nearest objects
    """

    conf = config.SingletonConfig.get_instance()
    similar_objects = faiss_db.search_similar(question)
    return [make_question_answer(qa_item) for qa_item in similar_objects]
