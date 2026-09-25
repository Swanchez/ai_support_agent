from ai_support_agent.mcp_cli import _build_legacy_display_context


def test_legacy_mcp_chunks_are_compacted_for_cli_display() -> None:
    context = _build_legacy_display_context(
        [
            {
                "source_id": "refund-policy-v1",
                "text": "## Срок\nДеньги поступают в течение 3–7 дней.",
            },
            {
                "source_id": "refund-policy-v1",
                "text": "## Условие\nСрок начинается после поступления товара на склад.",
            },
            {
                "source_id": "delivery-policy-v1",
                "text": "## Доставка\nСрок доставки указан в заказе.",
            },
        ]
    )

    assert context == (
        "Срок Деньги поступают в течение 3–7 дней.\n\n"
        "Доставка Срок доставки указан в заказе."
    )
