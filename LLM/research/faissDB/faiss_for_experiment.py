# faiss_for_experiment.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import yaml
from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Загрузка ресурсов NLTK
download('stopwords', quiet=True)
download('punkt', quiet=True)
stop_words = set(stopwords.words('russian'))

def preprocess_text(text: str) -> str:
    """
    Предобрабатывает текст: удаляет пунктуацию, приводит к нижнему регистру,
    удаляет стоп-слова и токенизирует текст.

    :param text: Исходный текст.
    :return: Предобработанный текст.
    """
    translator = str.maketrans('', '', string.punctuation)
    text = text.strip().translate(translator).lower()
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def convex_indexes(q_idx: int, counts: list[int]) -> (int, int):
    """
    Преобразует индекс FAISS в позиции элемента базы знаний и вопроса внутри элемента.

    :param q_idx: Индекс FAISS.
    :param counts: Список количества вопросов в каждом элементе базы знаний.
    :return: Кортеж (индекс элемента базы знаний, индекс вопроса внутри элемента).
    """
    elem_index = 0

    for item in counts:
        if q_idx < item:
            # Возвращаем элемент базы знаний, содержащий вопрос с номером q_idx
            # и номер вопроса внутри этого элемента
            return elem_index, q_idx

        q_idx -= item
        elem_index += 1

    raise ValueError("q_idx is out of bounds")

def initialize_model() -> SentenceTransformer:
    """
    Инициализирует и возвращает модель SentenceTransformer.

    :return: Объект модели SentenceTransformer.
    """
    transformer_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    model = SentenceTransformer(transformer_name)
    return model

def create_embeddings(model: SentenceTransformer, data: list[dict]) -> np.ndarray:
    """
    Создает и нормализует эмбеддинги для данных.

    :param model: Объект модели SentenceTransformer.
    :param data: Данные для создания эмбеддингов. Ожидается список словарей с ключами "question" и "answer".
    :return: Нормализованные эмбеддинги.
    """
    texts = [preprocess_text(item["question"] + " " + item["answer"]) for item in data]
    embeddings = model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    return embeddings

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Создает индекс FAISS для заданных эмбеддингов.

    :param embeddings: Эмбеддинги для индексации.
    :return: Объект индекса FAISS.
    """
    if embeddings.size == 0:
        raise ValueError("Эмбеддинги пусты. Невозможно создать индекс FAISS.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def search_similar(model: SentenceTransformer, index: faiss.IndexFlatIP, question: str, db_values: list[dict],
                  k_max: int, similarity_threshold: float) -> list[int]:
    """
    Ищет похожие ответы на заданный вопрос.

    :param model: Объект модели SentenceTransformer.
    :param index: Индекс FAISS.
    :param question: Входной вопрос.
    :param db_values: База данных для поиска. Ожидается список словарей с ключами "number", "question" и "answer".
    :param k_max: Максимальное количество результатов.
    :param similarity_threshold: Порог схожести.
    :return: Список номеров предсказанных ответов.
    """
    # Предобработка и кодирование вопроса
    preprocessed_question = preprocess_text(question)
    query_embedding = model.encode([preprocessed_question], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Поиск в FAISS
    distances, indices = index.search(np.array(query_embedding), k_max)

    # Получение количества вопросов в каждом элементе базы знаний
    elem_index_questions = [1 for item in db_values]

    predicted_numbers = []
    kb_items_idxes_set = set()

    for i in range(k_max):
        if distances[0][i] < similarity_threshold:
            break

        try:
            ans_pos, question_pos = convex_indexes(indices[0][i], elem_index_questions)
        except ValueError:
            continue  # Пропустить, если индекс выходит за пределы

        # Проверяем, был ли уже добавлен этот ответ
        if ans_pos not in kb_items_idxes_set:
            kb_items_idxes_set.add(ans_pos)
            predicted_numbers.append(db_values[ans_pos]["number"])

    return predicted_numbers
