# AI Support Agent

Учебный pet-проект: ИИ-ассистент поддержки с LLM, RAG, tool calling, агентом и MCP.

Проект построен по слоям: прикладная логика не зависит от интерфейса, к ядру уже подключены CLI, FastAPI browser UI и MCP-клиент.

## Что реализовано

- Gemini и OpenAI-клиенты за общим контрактом `LlmClient`; текущая практическая конфигурация использует Gemini.
- Структурированный ответ `SupportResponse`, проверяемый Pydantic: статус, ответ, альтернатива, рекомендации и источники.
- RAG по Markdown-политикам и PDF: chunking, embeddings Gemini, Qdrant или локальный JSON-индекс, метаданные и source-aware fallback для внешних справочных материалов.
- Оценка retrieval и ответов: Recall@k, Precision@k, негативные кейсы и проверка источников.
- Tools: список и статус заказов, отмена и заявка на возврат с подтверждением, валидация аргументов, ownership check и идемпотентность.
- Безопасный audit tool-вызовов: actor, tool, эффект и outcome без raw arguments, prompt-ов или секретов.
- Agent: planner, ограничение шагов, observations, read-tools, proposal для write-tool и HITL-подтверждение.
- MCP-сервер: tool поиска по базе знаний, resource с обзором сервиса и prompt-шаблон для поиска по политике.

## Архитектура

```text
CLI / FastAPI browser UI / MCP client
        |
AssistantService / AgentAssistantService
        |
  +-----+-------------------+
  |                         |
RAG retriever           Tool catalog/executor
  |                         |
Qdrant / JSON index   OrderRepository + ToolExecutionContext
  |
Gemini embeddings

Отдельный интеграционный контур:
MCP client <-> stdio MCP server <-> тот же RAG retriever
```

`service.py` собирает контекст для LLM и валидирует итоговый ответ

## Структура

```text
src/ai_support_agent/
  agents/       # planner, runtime, observations, evaluation и trace
  rag/          # документы, chunking, embeddings, индекс, retrieval, evaluation
  tools/        # каталог, executor, контекст, заказы, confirmation flow
  persistence/  # SQLAlchemy-модели, PostgreSQL-репозитории, seed-скрипты, audit adapter
  security/     # Argon2-хеширование паролей, JWT и authentication service
  web/          # FastAPI, browser UI, авторизация и история чатов
  mcp_server.py # MCP primitives и stdio entry point
  mcp_client.py # клиент, запускающий локальный MCP-сервер
  *_cli.py      # учебные консольные точки входа
knowledge/      # Markdown-политики и PDF-источники
data/           # JSON-индексы для offline-backend и первичного импорта в Qdrant
tests/          # unit- и интеграционные тесты без реальных API-вызовов
harness/        # отдельный учебный контур для coding-агента: политика, hooks, trace, evals
.agents/skills/ # процедуры coding-агента, включая ревью изменений
AGENTS.md       # инструкции coding-агенту по работе с репозиторием
```

## Установка

Нужен Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```


## Конфигурация и секреты

В `.env` указывается один активный LLM-провайдер и параметры embeddings:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```


### Локальная PostgreSQL через Docker

`compose.yaml` поднимает PostgreSQL 17 в контейнере `db`. Данные находятся в
именованном Docker volume `postgres_data`, а порт опубликован только на
`127.0.0.1:5432`, поэтому база не доступна из локальной сети.

```powershell
docker compose up -d db
docker compose ps
docker compose logs db
```

После статуса `healthy` можно открыть клиент PostgreSQL внутри контейнера:

```powershell
docker compose exec db psql -U ai_support_agent -d ai_support_agent
```

Проверка подключения из Python после установки зависимостей проекта:

```powershell
python -m ai_support_agent.persistence.check_connection
```

Миграции схемы хранятся в `migrations/` и управляются Alembic. После изменения
SQLAlchemy-моделей создаем черновик миграции, проверяем его и только затем применяем:

```powershell
alembic revision --autogenerate -m "создать таблицу заказов"
alembic upgrade head
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

Проверка тестов приложения:

```powershell
python -m pytest -q
```

### Web API, PostgreSQL и аутентификация

FastAPI запускается в режиме разработки так:

```powershell
uvicorn ai_support_agent.web.main:create_production_app --factory --reload
```

Страница входа доступна по адресу `http://127.0.0.1:8000/login`, а основной чат — по адресу `http://127.0.0.1:8000/`.
Swagger для разработчика остаётся на `http://127.0.0.1:8000/docs`.

В UI пользователь входит обычной формой и после этого попадает в единый чат
ассистента. Агент сам выбирает RAG-поиск или доступный tool для заказа;
отмена и заявка на возврат создаются только после явного подтверждения.
Для возврата пользователь указывает номер доставленного заказа и краткую
причину, затем передаёт товар в пункт приёма или отделение почты. Технический
JWT остаётся в `HttpOnly` cookie и не показывается в интерфейсе.

История чатов сохраняется только для текущего пользователя. Пустой «Новый чат»
не попадает в БД: запись создаётся с первым отправленным сообщением. Чат
удаляется после 24 часов неактивности, а в LLM передаются лишь последние 8
сообщений — это ограничивает рост контекста и расход токенов.


```powershell
docker compose up -d db qdrant
alembic upgrade head
python -m ai_support_agent.persistence.seed_demo_orders
python -m ai_support_agent.persistence.seed_demo_users
```

Для JWT в локальном `.env` нужны следующие настройки:

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
| `GET /api/v1/auth/session` | Вернуть безопасный идентификатор текущей сессии для UI | JWT/cookie |
| `POST /api/v1/chat` | Совместимый stateless-вызов единого агента | JWT/cookie |
| `GET /api/v1/conversations` | Список собственных чатов | JWT/cookie |
| `POST /api/v1/conversations` | Создать чат при первой отправке сообщения | JWT/cookie |
| `GET /api/v1/conversations/{conversation_id}/messages` | Прочитать сообщения собственного чата | JWT/cookie + ownership check |
| `POST /api/v1/conversations/{conversation_id}/messages` | Отправить сообщение в собственный чат | JWT/cookie + ownership check |
| `GET /api/v1/orders/{order_id}` | Получить только свой заказ | JWT/cookie + ownership check |
| `POST /api/v1/orders/{order_id}/cancellation` | Выполнить подтверждённую отмену | JWT/cookie + ownership + idempotency; cookie-вариант также CSRF |

Для тестовых аккаунтов после seed-скрипта доступны `demo-user-1` / `demo-password-1` и `demo-user-2` / `demo-password-2`. Это только локальные учебные данные.

В browser-варианте `/auth/login` выставляет две cookie: `support_access_token` с флагом `HttpOnly` и `support_csrf_token`. JavaScript UI не увидит JWT, но сможет передать CSRF-токен в `X-CSRF-Token` для write-запроса. Bearer-клиенты передают JWT явно и не нуждаются в CSRF-проверке.

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
2. Для чанков строятся embeddings. При `RAG_VECTOR_BACKEND=qdrant` они сохраняются в Qdrant вместе с текстом и метаданными; при `json` — в `data/rag_index.json` или `data/external_reference_rag_index.json`.
3. При вопросе создаётся embedding только вопроса; выбранный vector store возвращает ближайшие чанки выше порога cosine similarity.
4. В LLM передаются только найденные фрагменты, а `sources` заполняются приложением, а не моделью.

Для внутренней базы текущий проверенный порог — `0.70`: на JSON-backend и актуальном наборе
кейсов получены Recall@k = 1.00 и Precision@k = 1.00. Для внешней справочной
коллекции также используется `0.70`: полнота равна 1.00, а precision ниже,
потому что несколько близких юридических фрагментов могут быть полезны для
одного вопроса. Это не универсальная константа: при смене embedding-модели,
языка, структуры документов или коллекции порог нужно переоценивать.


### Qdrant через Docker Compose

Сервис `qdrant` использует закреплённый образ `qdrant/qdrant:v1.19.1`.
Данные находятся в volume `qdrant_data`, snapshots — в `qdrant_snapshots`.
REST API и dashboard опубликованы только на `127.0.0.1:6333`.

Добавь в локальный `.env`:

```dotenv
RAG_VECTOR_BACKEND=qdrant
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION_PREFIX=ai_support
QDRANT_TIMEOUT_SECONDS=10
```

Затем в активированном `.venv`:

```powershell
python -m pip install -e ".[dev]"
docker compose up -d qdrant
docker compose logs qdrant
Invoke-RestMethod http://127.0.0.1:6333/readyz
python -m ai_support_agent.rag.index_qdrant
python -m ai_support_agent.evaluate_retrieval --thresholds 0.70 --details
python -m ai_support_agent.evaluate_retrieval --collection external --thresholds 0.70 --details
```

Dashboard: `http://127.0.0.1:6333/dashboard`. Там видны коллекции,
количество points, векторы и payload с текстом и метаданными.
Для приложения, которое в будущем запускается внутри Compose, адрес будет
`http://qdrant:6333`.

Без `RAG_VECTOR_BACKEND` сохраняется прежний JSON-backend. Его также можно
явно выбрать значением `json`. Недоступность Qdrant считается ошибкой сервиса,
а не отсутствием релевантных знаний; HTTP API возвращает безопасный `503`.
`docker compose down` сохраняет данные, а `down -v` удаляет также volumes
Qdrant и PostgreSQL.

## Tools, agent и подтверждение

`ToolCatalog` — единый реестр tools, их JSON-схем, эффекта (`read`/`write`) и доступа агента. `ToolExecutor` валидирует аргументы, выполняет handler и не доверяет модели права доступа.

Заказы проверяются через `OrderRepository.find_visible_to(order_id, user_id)`. `ToolExecutionContext.current_user_id` создаётся приложением после авторизации и никогда не является аргументом, который придумывает LLM.

Read-tools `get_my_orders` и `get_order_status` возвращают только заказы текущего пользователя. В чате можно написать «Покажи мои заказы» или «Где заказ ORD-1001?»; модель выбирает инструмент, но не получает право подменить идентификатор пользователя.

`InMemoryPendingActionStore` фиксирует lifecycle write-действия в учебном диалоговом сценарии: `proposal_created`, `confirmation_approved`, `confirmation_rejected` или `confirmation_unclear`. После подтверждения `ToolExecutor` создаёт отдельное событие финального выполнения. Для production-композиции используется `PostgresAuditSink`, обёрнутый в `BestEffortAuditSink`: сбой аудита логируется, но не отменяет уже успешно выполненное действие. `InMemoryAuditSink` остаётся удобной реализацией для тестов.

`cancel_order` и `request_return` — write-tools: агент создаёт только
предложение действия, затем `ConversationService` ждёт явного подтверждения.
Возврат разрешён лишь для собственного заказа со статусом `delivered`; один
заказ может иметь только одну активную заявку. После подтверждения используется
idempotency key, чтобы повтор не выполнил действие второй раз.

## CI

`.github/workflows/tests.yml` запускает `pytest` в GitHub Actions на Ubuntu
при каждом push в `main` и при pull request в `main`. Тесты приложения и учебного
harness выполняются отдельными шагами; обычный `pytest` собирает только `tests/`.

## Учебный coding-agent harness

Отдельный контур для изучения безопасной и воспроизводимой работы coding-агента
с репозиторием. Это инструменты разработки, а не агент поддержки магазина:
они не участвуют в обработке пользовательских запросов веб-приложения.

- [AGENTS.md](AGENTS.md) задаёт правила работы с проектом; [skill ревью](.agents/skills/review-changes/SKILL.md)
  описывает повторяемую процедуру проверки. Делегированное ревью subagent
  продемонстрировано средствами coding-среды, не реализовано внутри runtime.
- `harness/command_policy.py` проверяет аргументы, применяет точный allowlist
  и возвращает `allow` / `ask` / `deny`.
- Учебный runtime фиксирует команду и рабочую папку, требует одноразового
  подтверждения для `ask`, ограничивает вызовы исполнителя и записывает события
  с общим идентификатором операции. `RecordingExecutor` только записывает вызовы
  в память: реальные shell-команды из этой демонстрации не запускаются.
- Отдельный адаптер `PreToolUse` проверяет shell-вызовы coding-среды до выполнения.
  В отличие от runtime, неизвестные строки он блокирует без собственного
  confirm-flow. Пользователь проверил блокировку в Codex CLI и расширении VS Code;
  временная локальная конфигурация после проверки отключена и исключена из Git.
- [Ручные evals](harness/evals/README.md) проверяют поиск ошибки, отсутствие ложного
  замечания и устойчивость к инструкции в недоверенном комментарии.
  [STATE.md](harness/STATE.md) фиксирует состояние и свидетельства проверок,
  но не переносит разрешения на будущие действия.

После установки dev-зависимостей запуск из корня проекта:

```powershell
# Демонстрация с подставным исполнителем, без реальных команд и LLM.
.\.venv\Scripts\python.exe -B -m harness.cli

# Отдельный набор тестов, без LLM, Docker и рабочей БД.
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider harness/tests
```

Тесты приложения и harness запускаются раздельно: совместный сбор с текущим
import mode конфликтует из-за одноимённых тестовых модулей. Прохождение pytest
не доказывает, что hook загружен в конкретной coding-среде — нужна отдельная
интеграционная проверка. Ручные evals агента используют реальные обращения к LLM.

Подробности: [демонстрация и ограничения](harness/README.md),
[подключение hook](harness/hooks/README.md), [перенос в другой проект](harness/PORTABILITY.md).
В репозитории хранится неактивный шаблон hook, без автоматической активации.
Строгий учебный allowlist блокирует почти все shell-команды; hook не заменяет
sandbox и не гарантирует блокировку при сбое собственного запуска. Бюджет
runtime не является лимитом токенов или стоимости coding-среды.

## MCP

MCP — внешний стандартизированный интерфейс, а не обязательная прокладка между модулями этого же приложения. Browser UI вызывает ядро напрямую; MCP пригодится внешним клиентам, IDE или другим host-приложениям.

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

## Текущие ограничения

- Заказы, идемпотентные результаты и audit-события уже хранятся в PostgreSQL. Pending-confirmation сценарий агента пока остаётся in-memory: для долгоживущих диалогов ему понадобится отдельное persistent-хранилище.
- HTTP API и browser UI используют JWT, CSRF-защиту cookie write-запросов и проверку владельца заказа. Это учебная локальная конфигурация; production потребует HTTPS, управления пользователями и секретами, rate limit и мониторинга.
- Локальный MCP использует `stdio`; удалённый многопользовательский MCP потребует HTTPS, проверку токенов и request-scoped context.
- RAG-корпус и demo-заказы невелики. В production понадобятся реальный Order API, расширенный набор документов, регулярные оценки retrieval и наблюдаемость.
