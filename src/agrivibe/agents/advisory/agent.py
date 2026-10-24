from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from agrivibe.agents.advisory.prompts import SYSTEM_PROMPT


def build_advisory_agent(model: BaseChatModel):
    """Builds the Advisory Agent: one grounded recommendation from the other agents' evidence.

    It has no tools. It does not fetch anything: everything it may use arrives in the
    message, taken from central memory.

    Args:
        model: The chat model that weighs the evidence and writes the recommendation.

    Returns:
        A compiled agent graph. Its last message is the recommendation.
    """
    return create_agent(model, tools=[], system_prompt=SYSTEM_PROMPT, name="advisory_agent")
