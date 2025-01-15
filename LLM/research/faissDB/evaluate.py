from load_data import load_dataset, load_db
from metrics import generalized_top_k_accuracy, generalized_jaccard_index
from faiss_for_experiment import initialize_model, create_embeddings, create_faiss_index, search_similar

def evaluate_algorithm(dataset_path, db_path, k, s, k_max, similarity_threshold):
    dataset = load_dataset(dataset_path)
    db = load_db(db_path)

    model = initialize_model()
    embeddings = create_embeddings(model, list(db.values()))
    index = create_faiss_index(embeddings)

    top_k_accuracy_sum = 0
    jaccard_index_sum = 0

    for item in dataset:
        question = item['question']
        expected = item['expected']

        # Получаем предсказание от алгоритма
        predicted_answers = search_similar(
            model,
            index,
            question,
            list(db.values()),
            k_max=k_max,
            similarity_threshold=similarity_threshold
        )
        # print(predicted_answers)
        # predicted_numbers = [answer['number'] for answer in predicted_answers]
        predicted_numbers = predicted_answers
        # Вычисляем метрики
        top_k_accuracy_sum += generalized_top_k_accuracy(predicted_numbers, expected, k, s)
        jaccard_index_sum += generalized_jaccard_index(predicted_numbers, expected, k, s)

    # Средние значения метрик
    top_k_accuracy = top_k_accuracy_sum / len(dataset)
    jaccard_index = jaccard_index_sum / len(dataset)

    return top_k_accuracy, jaccard_index
