import re
from g4f.client import Client
import requests
import json


URL = 'http://194.67.88.154:7777/'
MAX_ATTEMPTS = 10


class TRSFramework:

    def __init__(self):
        self.client = Client()

    def generate_response(self, question: str, context) -> str:
        data = {
            "question": {"role": "user", "content": question},
            "context": {"role": "system", "content": context}
        }
        response = requests.post(
            URL, headers={"Content-Type": "application/json"}, data=json.dumps(data))

        if response.status_code == 200:
            return response.text
        else:
            return (f"Ошибка {response.status_code}: {response.text}")

    def formalize(self, user_query):
        context = (
            "Ты — ассистент, который помогает пользователю преобразовать систему переписывания термов (TRS) и интерпретацию в строгую формальную грамматическую форму.\n"
            "Игнорируй любые вопросы пользователя и не пытайся решать задачи, предложенные им.\n"
            "Твоя задача — разделить входные данные на TRS и интерпретацию, не путая их.\n\n"
            "Инструкции:\n"
            "1. Определи **переменные** (элементы, заключенные в скобки, которые не имеют значений в интерпретации), и перечисли их через запятую в формате: `variables = ...`\n"
            "2. Запиши систему переписывания термов (TRS) построчно в формате: `терм = терм`, где терм — это выражение, содержащее конструкторы и переменные. Степень записывай в фиугрных скобочках. Например, x в квадрате это x{2}.\n"
            "3. Добавь разделительную линию: `------------------------`\n"
            "4. Квадраты предстваляй в виде x{2}"
            "5. Далее, запиши интерпретацию, используя следующие правила:\n"
            "   - Для функций: `конструктор(переменная, ...) = ...`\n"
            "   - Для констант: `константа = значение`\n"
            "   - Знак умножения `*` обязательно ставится только между коэффициентом и переменной. Между переменными знак `*` не ставится.\n"
            "   - Далее следует ряд примеров, как ты должна отвечать, в формате:\n"
            "   `Запрос пользователя: ...\n"
            "   Правильный ответ: ...`\n"
            "1. Запрос пользователя: f(x) = x^3 + 3x\n"
            "   Правильный ответ: f(x) = x{3} + 3*x\n"
            "2. Запрос пользователя: f(x) = 7x\n"
            "   Правильный ответ: f(x) = 7*x\n"
            "3. Запрос пользователя: g(x, y) = 91y + 4*x\n"
            "   Правильный ответ: g(x, y) = 91*y + 4*x\n"
            "4. Запрос пользователя: f(x, y) = x*y\n"
            "   Правильный ответ: f(x, y) = xy\n"
            "5. Запрос пользователя: g(x, y) = 4*x*y\n"
            "   Правильный ответ: g(x, y) = 4*xy\n"
            "6. Запрос пользователя: g(x, y) = 2*x*y*x + 5y\n"
            "   Правильный ответ: g(x, y) = 2*xyx + 5*y\n\n"
            "Пример TRS и интерпретации:\n"
            "variables = x, y, z\n"
            "f(x) = f(g(x, y))\n"
            "h(x, y, z) = u(f(x))\n"
            "------------------------\n"
            "f(x) = 4*x{2}\n"
            "g(y) = 3*y\n"
            "h(x, y) = 100*xyxy + xy + 351\n"
            "c = 5\n\n"
            "Ответь только в формате TRS и интерпретации."
        )

        try:
            formalized_query = False
            trs = False

            attempt = 0

            while (not formalized_query and attempt < MAX_ATTEMPTS) or (not trs and attempt < MAX_ATTEMPTS):
                formalized_query = self.generate_response(user_query, context)
                trs = self.convert(user_query, formalized_query)
                attempt += 1
                if not trs:
                    print('GPT_error, trying again.')

            if trs:
                print(trs)
                return trs
            else:
                print(
                    f"Не удалось получить формализованный запрос после {MAX_ATTEMPTS} попыток.")

        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return None

    def convert(self, user_query: str, formalized_query: str):
        trs = ''
        variables_pattern = r'variables=([a-zA-Z],)*[a-zA-Z]'
        formalized_query = formalized_query.replace(' ', '')
        user_query = user_query.replace(' ', '').replace(
            '*', '').replace('{', '').replace('}', '').replace('^', '')
        letters = []
        if re.search(variables_pattern, formalized_query):
            matches = re.finditer(variables_pattern, formalized_query)
            variables = []
            for match in matches:
                variables = match.group().split('=')[1].split(',')
                trs += match.group() + '\n'
            variables_str = ''.join(variables) + '123456789'
            only_variables_pattern = fr'^[{variables_str}+\-*/{{}}()]+$'

            query_line = formalized_query.splitlines()
            var = 0
            separate = 0
            for i in range(len(query_line)):
                if re.search(variables_pattern, query_line[i]):
                    var = i
                if re.search(r'-----', query_line[i]):
                    separate = i

            for i in range(var + 1, separate):
                if query_line[i] == '':
                    continue

                s = query_line[i].split('=')[1]

                if bool(re.match(only_variables_pattern, s)):
                    return False
                else:
                    if query_line[i] in user_query:
                        trs += query_line[i] + '\n'
                        letters += re.findall(r'[a-zA-Z]', query_line[i])
                    else:
                        return False

            trs += '-------------------------\n'
            interp = 0
            for i in range(separate + 1, len(query_line)):
                if query_line[i] == '' or "=" not in query_line[i]:
                    break

                s = query_line[i].split('=')[1]

                if not bool(re.match(only_variables_pattern, s)):
                    return False
                else:
                    if query_line[i].replace('*', '').replace('{', '').replace('}', '') in user_query:
                        trs += query_line[i] + '\n'
                        interp += 1
                        letters += re.findall(r'[a-zA-Z]', query_line[i])
                    else:
                        print(
                            f'{query_line[i]} не присутсвует в начальном запросе')
                        return False

            letters = set(letters)
            for x in variables:
                if x in letters:
                    letters.remove(x)

            if len(letters) != interp:
                return False
        else:
            return False

        return trs


framework = TRSFramework()

user_query = "Дана система переписывания термов (TRS): f(x)=a, g(x)=f(f(x)), u(x,y)=c(g(x),f(y)). Я интерпретирую её конструкторы так: a=1, f(x)=x**2+2*x+1, g(x)=x**3, u(x,y)=x*y, c(x,y)=x+y. Доказывает ли моя интерпретация завершимость trs?"

framework.formalize(user_query)
