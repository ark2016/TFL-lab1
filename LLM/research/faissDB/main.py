from evaluate import evaluate_algorithm
from faiss_for_experiment import initialize_model

if __name__ == "__main__":
    dataset_path = 'dataset.yaml'
    db_path = 'data_for_test.yaml'
    k = 5
    s = 1

    # Инициализация модели и токенизатора
    model, tokenizer, device = initialize_model()

    # Оценка качества работы алгоритма
    for similarity_threshold in [0.75, 0.73, 0.72, 0.71, 0.69, 0.68, 0.67, 0.66, 0.65]:
        for k_max in [3, 2]:
            try:
                top_k_accuracy, jaccard_index = evaluate_algorithm(
                    model,
                    tokenizer,
                    dataset_path,
                    db_path,
                    k,
                    s,
                    k_max=k_max,
                    similarity_threshold=similarity_threshold,
                    device=device
                )
                print(f"\n\nk_max: {k_max}, similarity_threshold: {similarity_threshold}")
                print(f"Generalized Top-k Accuracy: {str(top_k_accuracy).replace('.', ',')}")
                print(f"Generalized Jaccard Index: {str(jaccard_index).replace('.', ',')}")
            except IndexError as e:
                print(f"IndexError: {e}")
                print("Проверьте соответствие между эмбеддингами и данными.")
            except Exception as e:
                print(f"Произошла ошибка: {e}")
