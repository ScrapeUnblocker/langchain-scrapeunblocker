"""All ScrapeUnblocker tools in one toolkit, sharing one API key."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import Field, SecretStr

from langchain_scrapeunblocker._base import DEFAULT_BASE_URL
from langchain_scrapeunblocker.tools import (
    ScrapeUnblockerGoogleSearch,
    ScrapeUnblockerListPlugins,
    ScrapeUnblockerMarkdown,
    ScrapeUnblockerParsedData,
    ScrapeUnblockerRunPlugin,
    ScrapeUnblockerSearchAndRead,
)


class ScrapeUnblockerToolkit(BaseToolkit):
    """ScrapeUnblocker toolkit: read pages, search Google and run site plugins.

    Setup:
        .. code-block:: bash

            pip install -U langchain-scrapeunblocker
            export SCRAPEUNBLOCKER_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_scrapeunblocker import ScrapeUnblockerToolkit

            toolkit = ScrapeUnblockerToolkit()
            tools = toolkit.get_tools()
    """

    api_key: Optional[SecretStr] = Field(default=None)
    """ScrapeUnblocker API key. Read from ``SCRAPEUNBLOCKER_API_KEY`` (or
    ``SCRAPEUNBLOCKER_KEY``) when not passed."""

    base_url: str = DEFAULT_BASE_URL
    timeout: float = 180.0
    max_retries: int = 2
    include_plugins: bool = True
    """Add the list_plugins / run_plugin pair (100+ site scrapers)."""

    def get_tools(self) -> List[BaseTool]:
        common = {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.api_key is not None:
            common["api_key"] = self.api_key
        tools: List[BaseTool] = [
            ScrapeUnblockerMarkdown(**common),
            ScrapeUnblockerSearchAndRead(**common),
            ScrapeUnblockerGoogleSearch(**common),
            ScrapeUnblockerParsedData(**common),
        ]
        if self.include_plugins:
            tools += [ScrapeUnblockerListPlugins(**common), ScrapeUnblockerRunPlugin(**common)]
        return tools
