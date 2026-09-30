# AI Support Agent

Учебный pet-проект: ИИ-ассистент поддержки с LLM, RAG, tool calling, агентом и MCP.

Проект построен по слоям: прикладная логика не зависит от CLI, а внешние точки входа можно заменить на веб-интерфейс, API или MCP-клиент.

## Что уже реализовано

- Gemini и OpenAI-клиенты за общим контрактом `LlmClient`; текущая практическая конфигурация использует Gemini.
- Структурированный ответ `SupportResponse`, проверяемый Pydantic: статус, ответ, альтернатива, рекомендации и источники.
- RAG по Markdown-политикам и PDF: chunking, embeddings Gemini, локальный векторный индекс, метаданные и source-aware fallback для внешних справочных материалов.
- Оценка retrieval и ответов: Recall@k, Precision@k, негативные кейсы и проверка источников.
- Tools: статус заказа, отмена заказа с подтверждением, валидация аргументов, ownership check и идемпотентность.
- Безопасный audit tool-вызовов: actor, tool, эффект и outcome без raw arguments, prompt-ов или секретов.
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
  persistence/  # SQLAlchemy-модели, PostgreSQL-репозитории, seed-скрипты, audit adapter
  security/     # Argon2-хеширование паролей, JWT и authentication service
  web/          # FastAPI HTTP-адаптер и точка сборки web-приложения
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

### Локальная PostgreSQL через Docker

`compose.yaml` поднимает PostgreSQL 17 в контейнере `db`. Данные находятся в
именованном Docker volume `postgres_data`, а порт опубликован только на
`127.0.0.1:5432`, поэтому база не доступна из локальной сети.

Добавь в существующий локальный `.env` значения из блока `POSTGRES_*` файла
`.env.example`, заменив `POSTGRES_PASSWORD` на собственный пароль. Затем:

```powershell
docker compose up -d db
docker compose ps
docker compose logs db
```

После статуса `healthy` можно открыть клиент PostgreSQL внутри контейнера:

```powershell
docker compose exec db psql -U ai_support_agent -d ai_support_agent
```

Полезные команды `psql`: `\conninfo` — проверить подключение, `\dt` — список
таблиц, `\d имя_таблицы` — структура таблицы, `\q` — выход. Для диагностики
можно выполнять `SELECT`; изменение схемы вручную не используем — далее она
будет контролироваться миграциями Alembic.

`docker compose down` останавливает и удаляет контейнер, но сохраняет volume с
данными. `docker compose down -v` удаляет и volume: это полное удаление локальной
базы, применять его можно только осознанно.

Проверка подключения из Python после установки зависимостей проекта:

```powershell
python -m ai_support_agent.persistence.check_connection
```

Миграции схемы хранятся в `migrations/` и управляются Alembic. После изменения
SQLAlchemy-моделей создай черновик миграции, проверь его и только затем примени:

```powershell
alembic revision --autogenerate -m "создать таблицу заказов"
alembic upgrade head
```

После применения миграции можно отдельно наполнить **только локальную** БД
учебными заказами. Seed использует `ON CONFLICT DO NOTHING`, поэтому повторный
запуск не создаёт дубликаты и не перезаписывает существующие строки:

```powershell
python -m ai_support_agent.persistence.seed_demo_orders
```

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

### Web API, PostgreSQL и аутентификация

FastAPI запускается в режиме разработки так:

```powershell
uvicorn ai_support_agent.web.main:create_production_app --factory --reload
```

Swagger для разработчика доступен по адресу `http://127.0.0.1:8000/docs`.
Обычный пользователь в будущем будет работать через отдельный web-интерфейс, а не через Swagger.

Перед запуском подними локальную PostgreSQL и подготовь схему:

```powershell
docker compose up -d db
alembic upgrade head
python -m ai_support_agent.persistence.seed_demo_orders
python -m ai_support_agent.persistence.seed_demo_users
```

Для JWT в локальном `.env` нужны следующие настройки. Секрет генерируется один раз, не коммитится и не выводится в логи:

```dotenv
AUTH_JWT_SECRET=at-least-32-random-characters
AUTH_JWT_ISSUER=ai-support-agent
AUTH_ACCESS_TOKEN_TTL_MINUTES=30
AUTH_COOKIE_SECURE=false
```

`AUTH_COOKIE_SECURE=false` допустим только в локальной HTTP-разработке. При реальном HTTPS-развёртывании он обязан быть `true`.

| Endpoint | Назначение | Защита |
| --- | --- | --- |
| `GET /health` | Проверка доступности процесса | не требует аутентификации |
| `POST /api/v1/auth/token` | Получить Bearer JWT для Swagger или внешнего клиента | логин и пароль |
| `POST /api/v1/auth/login` | Browser-login с `HttpOnly` cookie | логин и пароль |
| `POST /api/v1/auth/logout` | Удалить browser cookie | JWT/cookie + CSRF |
| `POST /api/v1/chat` | Задать вопрос LLM/RAG | JWT/cookie |
| `GET /api/v1/orders/{order_id}` | Получить только свой заказ | JWT/cookie + ownership check |
| `POST /api/v1/orders/{order_id}/cancellation` | Выполнить подтверждённую отмену | JWT/cookie + ownership + idempotency; cookie-вариант также CSRF |

Для тестовых аккаунтов после seed-скрипта доступны `demo-user-1` / `demo-password-1` и `demo-user-2` / `demo-password-2`. Это только локальные учебные данные.

В browser-варианте `/auth/login` выставляет две cookie: `support_access_token` с флагом `HttpOnly` и `support_csrf_token`. JavaScript будущего UI не увидит JWT, но сможет передать CSRF-токен в `X-CSRF-Token` для write-запроса. Bearer-клиенты передают JWT явно и не нуждаются в CSRF-проверке.

Заказы, результаты идемпотентных операций и безопасные audit-события сохраняются в PostgreSQL. Audit не хранит prompt, аргументы tools, пароли, ключи или JWT. Если audit backend временно недоступен, `BestEffortAuditSink` записывает техническое предупреждение и не подменяет уже успешный результат операции ошибкой.

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

`InMemoryPendingActionStore` фиксирует lifecycle write-действия в учебном диалоговом сценарии: `proposal_created`, `confirmation_approved`, `confirmation_rejected` или `confirmation_unclear`. После подтверждения `ToolExecutor` создаёт отдельное событие финального выполнения. Для production-композиции используется `PostgresAuditSink`, обёрнутый в `BestEffortAuditSink`: сбой аудита логируется, но не отменяет уже успешно выполненное действие. `InMemoryAuditSink` остаётся удобной реализацией для тестов.

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

- Заказы, идемпотентные результаты и audit-события уже хранятся в PostgreSQL. Pending-confirmation сценарий агента пока остаётся in-memory: для долгоживущих диалогов ему понадобится отдельное persistent-хранилище.
- HTTP API использует JWT и проверяет владельца заказа. Следующим шагом будет пользовательский веб-интерфейс: он скроет технические токены за обычным входом в аккаунт.
- Локальный MCP использует `stdio`; удалённый многопользовательский MCP потребует HTTPS, проверку токенов и request-scoped context.
- Веб-интерфейс, реальная БД/Order API, авторизация и развёртывание будут следующими адаптерами поверх текущего ядра, а не переписыванием RAG/tools/agents с нуля.
