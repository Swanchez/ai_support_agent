"""Explicit model instructions for the current-order-status tool flow."""

from ai_support_agent.llm_client import LlmCall


ORDER_STATUS_TOOL_RULES = """Инструмент get_order_status предназначен только для получения актуального статуса
конкретного заказа. Вызывай его, только если пользователь спрашивает о текущем
статусе или доставке и явно указал номер формата ORD-<цифры>.
Если номера заказа нет, попроси пользователя сообщить его и не вызывай инструмент.
Не подставляй и не угадывай номер заказа. Если инструмент вернул order_not_found,
честно сообщи, что заказ с указанным номером не найден. Если он вернул временную
ошибку, сообщи о временной недоступности и предложи повторить попытку позже.

После результата инструмента сформируй ответ строго в согласованном JSON-формате."""


CANCEL_ORDER_TOOL_RULES = """Инструмент cancel_order предназначен только для отмены конкретного заказа.
Вызывай его только при явной просьбе отменить заказ с номером формата ORD-<цифры>.
Не сообщай об отмене до результата инструмента. Если инструмент вернул confirmation_required,
кратко попроси пользователя явно подтвердить отмену этого заказа."""


ORDER_TOOL_RULES = f"{ORDER_STATUS_TOOL_RULES}\n\n{CANCEL_ORDER_TOOL_RULES}"


ORDER_STATUS_SYSTEM_PROMPT = """Ты — AI Support Agent. Отвечай на русском языке, кратко и по делу.
Используй только факты из контекста и результатов инструментов. Не выдумывай данные.

""" + ORDER_STATUS_TOOL_RULES


def build_order_status_tool_call(user_question: str) -> LlmCall:
    """Build one tool-capable request without RAG context or hidden user identity data."""

    return LlmCall(
        messages=[
            {"role": "system", "content": ORDER_STATUS_SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ]
    )
