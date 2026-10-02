"""Composition helpers for a real bounded Gemini agent run."""

from ai_support_agent.agents.core import AgentPlanner, AgentRunner, AgentToolRegistry
from ai_support_agent.agents.gemini_planner import GeminiAgentPlanner
from ai_support_agent.agents.tools import (
    MyOrdersAgentTool,
    OrderStatusAgentTool,
    SearchKnowledgeBaseTool,
    agent_tool_definitions,
    proposal_agent_tool_names,
)
from ai_support_agent.config import GeminiConfig
from ai_support_agent.rag.retriever import Retriever
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.tools.executor import ToolExecutor


AGENT_MAX_STEPS = 3


def build_agent_runner(
    *,
    retriever: Retriever,
    planner: AgentPlanner,
    executor: ToolExecutor,
    context: ToolExecutionContext,
) -> AgentRunner:
    """Wire existing read capabilities into the provider-neutral agent core."""

    return AgentRunner(
        planner=planner,
        tools=AgentToolRegistry(
            {
                "search_knowledge_base": SearchKnowledgeBaseTool(retriever),
                "get_my_orders": MyOrdersAgentTool(executor, context),
                "get_order_status": OrderStatusAgentTool(executor, context),
            }
        ),
        max_steps=AGENT_MAX_STEPS,
    )


def build_gemini_agent_runner(
    *,
    retriever: Retriever,
    config: GeminiConfig,
    executor: ToolExecutor,
    context: ToolExecutionContext,
) -> AgentRunner:
    """Create Gemini agent with read actions and confirmation-gated proposals."""

    return build_agent_runner(
        retriever=retriever,
        planner=GeminiAgentPlanner(
            config,
            agent_tool_definitions(),
            proposal_tool_names=proposal_agent_tool_names(),
        ),
        executor=executor,
        context=context,
    )
