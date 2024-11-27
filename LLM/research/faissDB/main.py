from evaluate import evaluate_algorithm
from faiss_for_experiment import process_questions
def dummy_algorithm(question, db):
    # Пример заглушки алгоритма, который возвращает первые k элементов из БД
    return list(db.values())[:k]

def process_questions_eval(q, db):
    question = [{"question": q, "answer": ""}]
    print(process_questions(question, use_saved=True, filename="new_database"))

if __name__ == "__main__":
    dataset_path = 'dataset.yaml'
    db_path = 'data_for_test.yaml'
    k = 5
    s = 1

    top_k_accuracy, jaccard_index = evaluate_algorithm(dataset_path, db_path, process_questions_eval, k, s)
    print(f"Generalized Top-k Accuracy: {top_k_accuracy}")
    print(f"Generalized Jaccard Index: {jaccard_index}")


