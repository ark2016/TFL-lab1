from load_data import load_dataset, load_db
from metrics import generalized_top_k_accuracy, generalized_jaccard_index

def evaluate_algorithm(dataset_path, db_path, algorithm, k, s):
    dataset = load_dataset(dataset_path)
    db = load_db(db_path)

    top_k_accuracy_sum = 0
    jaccard_index_sum = 0

    for item in dataset:
        question = item['question']
        expected = item['expected']

        # Получаем предсказание от алгоритма
        predicted_answers = algorithm(question, db)
        predicted_numbers = [db[answer['number']]['number'] for answer in predicted_answers]

        # Вычисляем метрики
        top_k_accuracy_sum += generalized_top_k_accuracy(predicted_numbers, expected, k, s)
        jaccard_index_sum += generalized_jaccard_index(predicted_numbers, expected, k, s)

    # Средние значения метрик
    top_k_accuracy = top_k_accuracy_sum / len(dataset)
    jaccard_index = jaccard_index_sum / len(dataset)

    return top_k_accuracy, jaccard_index
