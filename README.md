# AI Support Agent

Учебный pet-проект: ИИ-ассистент поддержки с LLM, RAG, tool calling, агентом и MCP.

Проект построен по слоям: прикладная логика не зависит от CLI, а внешние точки входа можно заменить на веб-интерфейс, API или MCP-клиент.

## Что уже реализовано

- Gemini и OpenAI-клиенты за общим контрактом `LlmClient`; текущая практическая конфигурация использует Gemini.
- Структурированный ответ `SupportResponse`, проверяемый Pydantic: статус, ответ, альтернатива, рекомендации и источники.
- RAG по Markdown-политикам и PDF: chunking, embeddings Gemini, локальный векторный индекс, метаданные и source-aware fallback для внешних справочных материалов.
- Оценка retrieval и ответов: Recall@k, Precision@k, негативные кейсы и проверка источников.
- Tools: статус заказа, отмена заказа с подтверждением, валидация аргументов, ownership check и идемпотентность.
- Agent: planner, ограничение шагов, observations, read-tools, proposal для write-tool и HITL-подтверждение.
- MCP-сервер: tool поиска по базе знаний, resource с обзором сервиса и prompt-шаблон для поиска по политике.

## Архитектура

```text
CLI / будущий web UI
        |
AssistantService / AgentAssistantService
        |
  +-----+-------------------+
  |                         |
RAG retriever           Tool catalog/executor
  |                         |
vector index          OrderRepository + ToolExecutionContext
  |
Gemini embeddings

Отдельный интеграционный контур:
MCP client <-> stdio MCP server <-> тот же RAG retriever
```

`service.py` собирает контекст для LLM и валидирует итоговый ответ. Он не хранит знания, не знает деталей Gemini SDK и не управляет CLI.

## Структура

```text
src/ai_support_agent/
  agents/       # planner, runtime, observations, evaluation и trace
  rag/          # документы, chunking, embeddings, индекс, retrieval, evaluation
  tools/        # каталог, executor, контекст, заказы, confirmation flow
  mcp_server.py # MCP primitives и stdio entry point
  mcp_client.py # клиент, запускающий локальный MCP-сервер
  *_cli.py      # учебные консольные точки входа
knowledge/      # Markdown-политики и PDF-источники
data/           # кэшированные векторные индексы (не секреты)
tests/          # unit- и интеграционные тесты без реальных API-вызовов
```

## Установка

Нужен Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

В VS Code выбери интерпретатор из `.venv`. Это важно: CLI, тесты и дочерний MCP-процесс должны использовать одинаковое виртуальное окружение.

## Конфигурация и секреты

В `.env` указывается один активный LLM-провайдер и параметры Gemini embeddings:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```

`.env` остаётся только на локальной машине и не коммитится. Не помещай ключи в исходный код, README, логи или скриншоты.

LLM-вызовы и создание embeddings могут расходовать квоту провайдера. Unit-тесты используют заглушки и не должны обращаться к API; ручные CLI-команды и evaluation-команды — могут.

## Основные команды

Запуск основного диалога:

```powershell
python -m ai_support_agent.cli
```

Запуск agent-версии с безопасным trace:

```powershell
python -m ai_support_agent.agent_cli --debug
```

Проверка всех тестов:

```powershell
python -m pytest -q
```

### RAG и оценки

```powershell
# Подробная оценка internal RAG на выбранном пороге.
python -m ai_support_agent.evaluate_retrieval --thresholds 0.70 --details

# Оценка retrieval по внешним справочным PDF.
python -m ai_support_agent.evaluate_retrieval --collection external --details

# Проверка итоговых ответов и траекторий агента.
python -m ai_support_agent.evaluate_answers
python -m ai_support_agent.evaluate_agents

# Предпросмотр извлечённого текста PDF перед индексацией.
python -m ai_support_agent.inspect_pdf knowledge/pdf/remote_sales_return.pdf
```

Первые две команды оценки создают embeddings; последние две также могут вызывать LLM.

## RAG

Внутренние правила находятся в `knowledge/*.md`, внешние справочные документы — в `knowledge/pdf/`.

При создании или изменении базы знаний:

1. Документы преобразуются в чанки с `document_id`, `chunk_id`, типом источника и номером страницы при наличии.
2. Для чанков один раз строятся embeddings и сохраняются в `data/rag_index.json` или `data/external_reference_rag_index.json`.
3. При вопросе создаётся embedding только вопроса; локальный vector store возвращает ближайшие чанки выше порога.
4. В LLM передаются только найденные фрагменты, а `sources` заполняются приложением, а не моделью.

Для внутренней базы текущий проверенный порог — `0.70`. Это не универсальная константа: при смене embedding-модели, языка, структуры документов или коллекции его нужно переоценивать на наборе кейсов.

## Tools, agent и подтверждение

`ToolCatalog` — единый реестр tools, их JSON-схем, эффекта (`read`/`write`) и доступа агента. `ToolExecutor` валидирует аргументы, выполняет handler и не доверяет модели права доступа.

Заказы проверяются через `OrderRepository.find_visible_to(order_id, user_id)`. `ToolExecutionContext.current_user_id` создаётся приложением после авторизации и никогда не является аргументом, который придумывает LLM.

`cancel_order` — write-tool: агент создаёт только предложение действия, затем `ConversationService` ждёт явного подтверждения. После подтверждения используется idempotency key, чтобы повтор не отменил заказ второй раз.

## MCP

MCP — внешний стандартизированный интерфейс, а не обязательная прокладка между модулями этого же приложения. Основной web UI в будущем будет вызывать ядро напрямую; MCP пригодится внешним клиентам, IDE или другим host-приложениям.

### Доступные primitives

| Тип | Идентификатор | Назначение |
| --- | --- | --- |
| Tool | `search_knowledge_base(query)` | Ищет релевантные фрагменты политик через существующий RAG. |
| Resource | `support://service/overview` | Возвращает безопасный обзор возможностей и ограничений сервера. |
| Prompt | `support-policy-answer(question)` | Возвращает шаблон сценария поиска; сам не вызывает RAG и LLM. |

Проверить MCP-контур:

```powershell
# Discovery и один поиск.
python -m ai_support_agent.mcp_cli

# Прочитать статичный resource.
python -m ai_support_agent.mcp_cli --overview

# Получить prompt-шаблон, без обращения к LLM/RAG.
python -m ai_support_agent.mcp_cli --policy-prompt "Когда придут деньги за возврат?"

# Временная диагностика MCP retrieval.
python -m ai_support_agent.mcp_cli --debug
```

### Как работает stdio

`mcp_cli` создаёт `SupportMcpClient`. Тот запускает дочерний процесс тем же интерпретатором Python:

```text
python -m ai_support_agent.mcp_cli
  -> python -m ai_support_agent.mcp_server
  -> MCP JSON-RPC через stdin/stdout
```

Один блок `async with client.connect()` держит соединение и дочерний сервер живыми для discovery и последующего вызова tool. После завершения CLI процесс сервера закрывается.

`stdout` MCP-сервера зарезервирован для JSON-RPC. Обычный `print()` в `mcp_server.py` сломает протокол; ошибки старта идут в `stderr` через `SystemExit`, а отладочные данные возвращаются только opt-in полем tool-ответа.

Обычный MCP-сервер публикует только read-only RAG. Фабрика `create_authenticated_support_mcp_server(...)` добавляет `get_order_status(order_id)` лишь если доверенный host передал `ToolExecutionContext`. Она демонстрирует границу авторизации; для многопользовательского HTTP-варианта контекст должен создаваться на каждый проверенный запрос/сеанс, а не храниться глобально.

## Текущие ограничения и следующий этап

- Данные заказов и pending confirmations пока хранятся в памяти и предназначены для обучения.
- `DEMO_TOOL_CONTEXT` не является настоящей пользовательской аутентификацией.
- Локальный MCP использует `stdio`; удалённый многопользовательский MCP потребует HTTPS, проверку токенов и request-scoped context.
- Веб-интерфейс, реальная БД/Order API, авторизация и развёртывание будут следующими адаптерами поверх текущего ядра, а не переписыванием RAG/tools/agents с нуля.
