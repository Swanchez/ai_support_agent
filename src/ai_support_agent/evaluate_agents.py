"""Manual live evaluation of agent trajectories and their final answers."""

from ai_support_agent.agents.evaluation import (
    DEFAULT_AGENT_EVALUATION_CASES,
    AgentEvaluationResult,
    evaluate_agent_run,
)
from ai_support_agent.agents.runtime import build_gemini_agent_runner
from ai_support_agent.config import load_gemini_config
from ai_support_agent.exceptions import (
    ConfigurationError,
    EmbeddingRequestError,
    InvalidModelResponseError,
    LlmRequestError,
)
from ai_support_agent.rag.fallback_retriever import FallbackRetriever
from ai_support_agent.rag.runtime import (
    create_external_reference_gemini_vector_retriever,
    create_gemini_vector_retriever,
)
from ai_support_agent.service import EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT
from ai_support_agent.tools.executor import DEFAULT_TOOL_EXECUTOR


def create_agent_retriever() -> FallbackRetriever:
    """Use the same retrieval policy as the agent's manual console command."""

    return FallbackRetriever(
        primary=create_gemini_vector_retriever(),
        fallback_factory=create_external_reference_gemini_vector_retriever,
        fallback_threshold=EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD,
    )


def format_result(result: AgentEvaluationResult) -> str:
    """Render actions, sources, limits, and precise deterministic failures."""

    run = result.run
    actions = ", ".join(
        observation.action.name for observation in run.state.observations
    ) or "nothing"
    sources = ", ".join(run.response.sources) or "nothing"
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"[{status}] {result.case.question}",
        f"  actions={actions}",
        f"  status={run.response.status.value}; sources={sources}",
        f"  steps={run.state.step_count}; limit_reached={run.step_limit_reached}",
        f"  answer={run.response.answer}",
    ]
    lines.extend(f"  failure={failure}" for failure in result.failures)
    return "\n".join(lines)


def main() -> None:
    """Run each live scenario once; each one can call embeddings and Gemini."""

    try:
        retriever = create_agent_retriever()
        runner = build_gemini_agent_runner(
            retriever=retriever,
            config=load_gemini_config(),
            executor=DEFAULT_TOOL_EXECUTOR,
            context=DEMO_TOOL_CONTEXT,
        )
        for case in DEFAULT_AGENT_EVALUATION_CASES:
            run = runner.run(case.question)
            print(format_result(evaluate_agent_run(case, run)))
    except ConfigurationError as error:
        print(f"Configuration error: {error}")
    except (LlmRequestError, EmbeddingRequestError):
        print("Could not call the configured model or embedding service.")
    except InvalidModelResponseError:
        print("The model returned an invalid agent response.")


if __name__ == "__main__":
    main()
