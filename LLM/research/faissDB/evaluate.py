from load_data import load_dataset, load_db
from metrics import generalized_top_k_accuracy, generalized_jaccard_index
from faiss_for_experiment import initialize_model, create_embeddings, create_faiss_index, search_similar

def evaluate_algorithm(model, tokenizer, dataset_path, db_path, k, s, k_max, similarity_threshold, device):
    """
    Оценивает алгоритм на основе заданного набора данных и базы данных.

    Args:
        model: Загруженная модель.
        tokenizer: Загруженный токенизатор.
        dataset_path: Путь к файлу с набором данных для оценки.
        db_path: Путь к файлу с базой данных.
        k: Параметр для метрики top-k.
        s: Параметр для метрики.
        k_max: Максимальное количество результатов для поиска.
        similarity_threshold: Порог схожести для фильтрации результатов.
        device: Устройство для вычислений.

    Returns:
        top_k_accuracy: Средняя метрика top-k accuracy.
        jaccard_index: Средняя метрика Jaccard index.
    """
    dataset = load_dataset(dataset_path)
    db = load_db(db_path)

    data_list = list(db.values())
    embeddings = create_embeddings(model, tokenizer, data_list, device)
    index = create_faiss_index(embeddings, len(data_list))

    if index.ntotal != len(data_list):
        print(f"Error: FAISS index содержит {index.ntotal} эмбеддингов, но список данных содержит {len(data_list)} элементов.")
        raise ValueError("Несоответствие между количеством эмбеддингов в FAISS индексе и количеством элементов в данных.")

    top_k_accuracy_sum = 0
    jaccard_index_sum = 0

    for item in dataset:
        question = item['question']
        expected = item['expected']

        # Получаем предсказание от алгоритма
        predicted_answers = search_similar(
            model,
            tokenizer,
            index,
            question,
            data_list,
            device,
            k_max=k_max,
            similarity_threshold=similarity_threshold
        )
        predicted_numbers = [answer['number'] for answer in predicted_answers]

        # Вычисляем метрики
        top_k_accuracy_sum += generalized_top_k_accuracy(predicted_numbers, expected, k, s)
        jaccard_index_sum += generalized_jaccard_index(predicted_numbers, expected, k, s)

    # Средние значения метрик
    top_k_accuracy = top_k_accuracy_sum / len(dataset) if len(dataset) > 0 else 0
    jaccard_index = jaccard_index_sum / len(dataset) if len(dataset) > 0 else 0

    return top_k_accuracy, jaccard_index
