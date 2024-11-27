from evaluate import evaluate_algorithm

if __name__ == "__main__":
    dataset_path = 'dataset.yaml'
    db_path = 'data_for_test.yaml'
    k = 5
    s = 1

    # Оценка качества работы алгоритма
    top_k_accuracy, jaccard_index = evaluate_algorithm(dataset_path, db_path, k, s)
    print(f"Generalized Top-k Accuracy: {top_k_accuracy}")
    print(f"Generalized Jaccard Index: {jaccard_index}")
