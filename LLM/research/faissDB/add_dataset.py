from faiss_for_experiment import *
import yaml

file_path = '../../../data/data.yaml'

with open(file_path, 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

# Удаление поля 'author' из каждого элемента
filtered_data = [{k: v for k, v in item.items() if k != 'author'} for item in data]

questions_list = filtered_data

# Создание новой базы данных
results = process_questions(questions_list, use_saved=False, filename="new_database")

# Вывод результатов
for result in results:
    print(result)
    print("=" * 50)
