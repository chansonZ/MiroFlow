# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
IO processor base class

IO processors are specialized Agents for handling input and output.
They inherit from BaseAgent but serve specific purposes.
"""

from miroflow.agents.base import BaseAgent


class BaseIOProcessor(BaseAgent):
    """
    IO processor base class

    IO processors are used for:
    - Input processing: generating prompts, handling user input
    - Output processing: generating summaries, extracting final answers

    IO processors do not call external tools, so MCP/tool initialization is
    skipped automatically (``USE_TOOLS = False``).  Individual subclasses that
    exceptionally need tool access can override ``USE_TOOLS = True``.
    """

    USE_TOOLS: bool = False
