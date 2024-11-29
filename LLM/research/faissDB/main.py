from evaluate import evaluate_algorithm

if __name__ == "__main__":
    dataset_path = 'dataset.yaml'
    db_path = 'data_for_test.yaml'
    k = 5
    s = 1

    # Оценка качества работы алгоритма
    for k_max in [5, 3, 2, 1]:
        for similarity_trashold in [0.8, 0.75, 0.72, 0.7, 0.65, 0.6]:
            top_k_accuracy, jaccard_index = evaluate_algorithm(dataset_path, db_path, k, s, k_max=k_max,
                                                               similarity_trashold=similarity_trashold)
            print(f"\n\nk_max: {k_max}, similarity_trashold: {similarity_trashold}")
            print(f"Generalized Top-k Accuracy: {str(top_k_accuracy).replace('.', ',')}")
            print(f"Generalized Jaccard Index: {str(jaccard_index).replace('.', ',')}")
