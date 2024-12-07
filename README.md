# rofl-lab1

## Гайд на отправку запросов к LLM

### Пререквизиты
- склонирован репозиторий
- установлен [Docker](https://www.docker.com)

1) Собираем `Docker` образ:
    ```bash
    docker build -f ./dockerfiles/llm.dockerfile -t my-t .
    ```
2) Запускаем `Docker` контейнер
    ```bash
   # подставить API ключ mistral
    docker run -p 8100:8100 -e MISTRAL_API_KEY=<API_КЛЮЧ_MISTRAL> my-t
    ```
3) Переходим на http://localhost:8100/docs. Это раздел с документацией к `API`, отсюда можно отправлять
запросы на это самое `API`. Делается это так: выбираем любую ручку (например `/ping`), кликаем по ней, там 
будет кнопка `Try it out`, нажимаем туда - появляется поле ввода для всех параметров, вводим и кликаем на кнопку
`Execute`. У каждой ручки также есть описание, что она делает. Для поиска похожих для вопроса надо пользоваться
[/search_similar](http://0.0.0.0:8100/docs#/Questions/api_search_similar_search_similar_post), а для отправки запроса 
в `mistral` - [/get_chat_response](http://0.0.0.0:8100/docs#/Questions/api_get_chat_response_get_chat_response_post).

**Важно**

Не забудьте указать версию модели, у нас она все еще не зафиксирована,
так что используйте `mistral-large-latest`. Если модель не будет указана,
то по умолчанию будет использоваться `open-mistral-7b`, она очевидно
будет "поглупее" и вы можете найти галлюцинации, которых не будет на
"проде".

## Автотесты для поиска близких

У нас появились автотесты для поиска близких. Добавить свой тест можно
в файле [tests.yaml](/LLM/test/tests.yaml). Пока что поддерживается лишь 
один формат тестов: проверка на вхождение в контекст.
Формат теста:

```yaml
# тут должен стоять вопрос, для которого будем искать 
# ближайшие элементы базы знаний
- question: Что такое счетчиковая машина?
  # тут должны стоять вопросы, которые ожидаются, что будут в
  # списке ближайших. Если хотя бы одного не будет - 
  # тест упадет и покажет ошибку
  should_include:
    - Что такое счетчиковая машина?

```

Тесты можно запускать как локально, так и в `CI`, во втором
случае они будут запускаться автоматически.

## Локальный запуск тестов

1) Сборка образа
   ```bash
   docker build -f ./dockerfiles/similarity-tester.dockerfile -t my-t .
   ```
2) Запуск контейнера
   ```bash
   docker run my-t
   ```
