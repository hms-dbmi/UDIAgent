"""Backward-compatibility shim — imports from the new package location."""
from udiagent.agent import *  # noqa: F401, F403
from udiagent.agent import UDIAgent, _make_openai_client  # noqa: F401
