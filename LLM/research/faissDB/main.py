from evaluate import evaluate_algorithm

if __name__ == "__main__":
    dataset_path = 'dataset.yaml'
    db_path = 'data_for_test.yaml'
    k = 5
    s = 1

    # Оценка качества работы алгоритма
    for k_max in [100]:
        for similarity_threshold in [0.5]:
            top_k_accuracy, jaccard_index = evaluate_algorithm(dataset_path, db_path, k, s, k_max=k_max,
                                                               similarity_threshold=similarity_threshold)
            print(f"\n\nk_max: {k_max}, similarity_trashold: {similarity_threshold}")
            print(f"Generalized Top-k Accuracy: {str(top_k_accuracy).replace('.', ',')}")
            print(f"Generalized Jaccard Index: {str(jaccard_index).replace('.', ',')}")
