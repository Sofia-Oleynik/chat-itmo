import os
import uvicorn
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.agents import create_react_agent
from langchain.llms import BaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from urllib.parse import urljoin
import torch

api_key = "ok"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
model_name = "google/mt5-small"

# Загружаем переменные окружения
load_dotenv()

app = FastAPI()

# Определение модели запроса
class RequestModel(BaseModel):
    query: str
    id: int


class ClassLLM(BaseLLM):

    def _llm_type(self) -> str:
        return model_name

    def _generate(self, prompts: list) -> list:
        # URL API
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {api_key}"}

        responses = []

        for prompt in prompts:
            response = requests.post(url, headers=headers, json={"inputs": prompt})

            if response.status_code == 200:
                generated_text = response.json()
                responses.append(generated_text)
            else:
                raise Exception(f"Error: {response.status_code}, Message: {response.text}")

        return responses

llm = ClassLLM(model_name=model_name, api_key=api_key)
# Создание инструмента поиска DuckDuckGo
search_tool = DuckDuckGoSearchRun()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}

# Пример пользовательского инструмента для извлечения информации из официальных источников ИТМО
def custom_tool(query):
    # URL официальных страниц ИТМО (примерные ссылки, замените на актуальные)
    base_domain = "https://itmo.ru"
    subdomains = [
        "en",
        "ru",
        "news",
        "projects",
        "education",
        "contacts",
        "science"
    ]
    # Наддомены, которые могут быть использованы
    superdomains = [
        "www.itmo.ru",
        "student.itmo.ru",
        "research.itmo.ru",
        "org.itmo.ru",
        "news.itmo.ru"
    ]

    # Генерация списка URL, включая поддомены и наддомены
    sources = [urljoin(base_domain, f"{sub}/") for sub in subdomains] + [f"https://{sd}" for sd in superdomains]

    all_parsed_data = []

    for url in sources:
        try:
            # Выполняем GET-запрос к официальным страницам ИТМО
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Передаём HTML-код страницы в OpenAI для парсинга
                parsing_response = llm._generate([response.text])

                all_parsed_data.append(parsing_response)  # Добавляем результат парсинга

                print(f"Retrieved and parsed content from {url}.")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")

    # Объединяем все парсинг ответы
    combined_data = " ".join(all_parsed_data)

    if combined_data:
        return combined_data
    else:
        return "Не удалось получить информацию из официальных источников."


custom_tool_instance = Tool(
    name="CustomTool",
    description="Эта функция извлекает информацию из различных официальных источников ИТМО.",
    func=custom_tool
)

# Создание промта для агента
tools = [search_tool, custom_tool_instance]

template = """# Вы — выдающийся AI-агент, который имеет доступ к дополнительным инструментам для ответа на вопросы.

<agent_tools>
Ответьте на следующие вопросы наилучшим образом. У вас есть доступ к следующим инструментам:

{tools}

</agent_tools>

Пользовательский ввод: {input}

<agent_instruction>
# **Инструкции для агента:**
Пользователь задаёт вопрос, содержащий варианты ответов, пронумерованные цифрами от 1 до 10. Каждый вариант соответствует определённому утверждению или факту. 
Вы должны определить правильный вариант ответа и вернуть его в поле answer JSON-ответа. Если вопрос не подразумевает выбор из вариантов, поле answer должно содержать null.

## Важно:
- Если заданный вопрос не связан с деятельностью или информацией о ИТМО, вы должны отказаться от ответа. В этом случае просто ответьте: "Извините, я могу отвечать только на вопросы, связанные с ИТМО."

## Структура вопроса:
- Вопрос всегда начинается с текстового описания.
- После описания перечисляются варианты ответов, каждый из которых пронумерован цифрой от 1 до 10.
- Варианты ответов разделяются символом новой строки (\\n).

## Ответ бота:
- Все ответы формируются и возвращаются в формате JSON со следующими ключами:
  - **id** — числовое значение, соответствующее идентификатору запроса (передаётся во входном запросе).
  - **answer** — числовое значение, содержащее правильный ответ на вопрос (если вопрос подразумевает выбор из вариантов). Если вопрос не подразумевает выбор из вариантов, значение должно быть null.
  - **reasoning** — текстовое поле, содержащее объяснение или дополнительную информацию по запросу.
  - **sources** — список ссылок на источники информации (если используются). Если источники не требуются, значение должно быть пустым списком [].

- Бот должен определить правильный вариант ответа.
- Правильный вариант указывается в поле answer (например, 1, 2, ..., 10).
- Если вопрос не предполагает выбор из вариантов, поле answer должно быть null.

## Дополнительная информация:
- **Не забывайте использовать инструменты, если это необходимо, для получения наилучшего ответа.**
</agent_instruction>

# Используйте следующий формат:
Проанализируйте вопрос и используйте следующий формат для ответа.

Question: {input}
Thought: Могу ли я использовать инструменты? Если да, то какие?
Action: Действие, которое нужно предпринять, должно быть одним из [{tool_names}]
Action Input: Ввод для действия (не используйте "```json")
Observation: Результат действия
... (эти Thought/Action/Action Input/Observation могут повторяться N раз)


Подтвердите, что вы понимаете инструкцию!

Вопрос: {input}
Thought: {agent_scratchpad}
"""

prompt_template = PromptTemplate.from_template(template)

# Создание агента с инструментами
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)


@app.post("/api/request")
async def handle_request(request_model: RequestModel):
    query = request_model.query
    request_id = request_model.id
    # Формирование сообщения для агента
    message = prompt_template.format(input=query, tools=[tool.name for tool in tools],
                                     tool_names=', '.join([tool.name for tool in tools]))

    # Выполнение агента с заданным запросом
    answer = agent({"input": message})

    # Формируем ответ
    response = {
        "id": request_id,
        "answer": answer.get("answer", None),
        "reasoning": f"Ответ сгенерирован с помощью модели {model_name}",
        "sources": [result['link'] for result in answer.get("search_results", [])]
    }

    return response

if __name__ == '__main__':
    print(llm._generate(['Расскажи про ИТМО']))
    uvicorn.run(app, host="0.0.0.0", port=5000)
