import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
import faiss
import numpy as np


def initialize_model():
    try:
        # Загрузка конфигурации с обработкой отсутствующего атрибута
        config = AutoConfig.from_pretrained('ai-sage/Giga-Embeddings-instruct', trust_remote_code=True)

        if not hasattr(config.latent_attention_config, '_attn_implementation_internal'):
            config.latent_attention_config._attn_implementation_internal = None

        # Загрузка модели с исправленной конфигурацией
        model = AutoModel.from_pretrained('ai-sage/Giga-Embeddings-instruct', config=config, trust_remote_code=True)

        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained('ai-sage/Giga-Embeddings-instruct', trust_remote_code=True)

        # Перенос модели на GPU, если доступно
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        return model, tokenizer, device
    except Exception as e:
        print(f"Ошибка при инициализации модели или токенизатора: {e}")
        raise


def create_embeddings(model, tokenizer, data, device):
    """
    Создаёт эмбеддинги для объединённых вопросов и ответов.

    Args:
        model: Загруженная модель.
        tokenizer: Загруженный токенизатор.
        data: Список словарей с ключами "question" и "answer".
        device: Устройство для вычислений.

    Returns:
        NumPy массив эмбеддингов.
    """
    # Объединение вопроса и ответа в одну строку
    combined_texts = [f"Question: {item['question']} Answer: {item['answer']}" for item in data]

    # Инструкция, если требуется моделью
    instruction = "Generate embeddings for the following Q&A pairs."

    # Получение эмбеддингов
    inputs = tokenizer(combined_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Предполагается, что эмбеддинг берётся из последнего скрытого состояния
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # Нормализация эмбеддингов
    embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1).numpy()

    return embeddings


def create_faiss_index(embeddings, data_length):
    """
    Создаёт FAISS индекс для заданных эмбеддингов.

    Args:
        embeddings: NumPy массив эмбеддингов.
        data_length: Количество элементов в данных.

    Returns:
        FAISS индекс.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Используем Inner Product (cosine similarity после нормализации)
    faiss.normalize_L2(embeddings)  # Нормализуем эмбеддинги
    index.add(embeddings)
    print(f"FAISS index содержит {index.ntotal} эмбеддингов.")
    print(f"Список данных содержит {data_length} элементов.")
    return index


def search_similar(model, tokenizer, index, query, data, device, k_max=2, similarity_threshold=0.69):
    """
    Ищет похожие элементы в FAISS индексе для заданного запроса.

    Args:
        model: Загруженная модель.
        tokenizer: Загруженный токенизатор.
        index: FAISS индекс.
        query: Вопрос для поиска.
        data: Список словарей с ключами "question" и "answer".
        device: Устройство для вычислений.
        k_max: Максимальное количество результатов для поиска.
        similarity_threshold: Порог схожести для фильтрации результатов.

    Returns:
        Список похожих элементов.
    """
    # Объединение запроса с инструкцией
    combined_query = f"Question: {query} Answer:"

    # Получение эмбеддинга для запроса
    inputs = tokenizer([combined_query], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # Нормализация эмбеддинга
    query_embedding = F.normalize(torch.tensor(query_embedding), p=2, dim=1).numpy()

    # Поиск в FAISS
    D, I = index.search(query_embedding, k_max)
    print(f"Результаты поиска для запроса '{query}': Индексы={I}, Расстояния={D}")

    similar_items = []
    for idx, distance in zip(I[0], D[0]):
        if distance >= similarity_threshold and idx < len(data):
            similar_items.append({"distance": distance, "item": data[idx]})
            print(f"Distance: {distance}, Question: {data[idx]['question']}")
        else:
            print(f"Warning: idx {idx} is out of range или distance {distance} ниже порога.")

    return [item["item"] for item in similar_items]
