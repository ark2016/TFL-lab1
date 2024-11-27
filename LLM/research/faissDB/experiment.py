from faiss_for_experiment import *
import yaml

file_path = 'dataset.yaml'

with open(file_path, 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# questions = [data[i]['question'] for i in range(0, len(data))]

# results = process_questions(questions, use_saved=True, filename="new_database")
for i in range(0, len(data)):
    question = [{"question": data[i]['question'], "answer": ""}]
    print(process_questions(question, use_saved=True, filename="new_database"))
    print("\n\n\n\n")
