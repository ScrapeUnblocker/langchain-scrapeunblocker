"""LangChain's standard integration tests, against the real API.

Needs SCRAPEUNBLOCKER_API_KEY; SCRAPEUNBLOCKER_BASE_URL points them at another
API host (e.g. a staging one). Run: pytest tests/integration_tests
"""

import os
from typing import Any, Type

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_scrapeunblocker import (
    ScrapeUnblockerGoogleSearch,
    ScrapeUnblockerMarkdown,
    ScrapeUnblockerParsedData,
    ScrapeUnblockerRunPlugin,
    ScrapeUnblockerSearchAndRead,
)

pytestmark = pytest.mark.skipif(
    not (os.environ.get("SCRAPEUNBLOCKER_API_KEY") or os.environ.get("SCRAPEUNBLOCKER_KEY")),
    reason="needs SCRAPEUNBLOCKER_API_KEY",
)


class _Base(ToolsIntegrationTests):
    @property
    def tool_constructor_params(self) -> dict:
        base_url = os.environ.get("SCRAPEUNBLOCKER_BASE_URL")
        return {"base_url": base_url} if base_url else {}


class TestMarkdownIntegration(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerMarkdown

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": "https://example.com"}


class TestParsedDataIntegration(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerParsedData

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"}


class TestGoogleSearchIntegration(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerGoogleSearch

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "langchain"}


class TestSearchAndReadIntegration(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerSearchAndRead

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "python requests library", "results": 1}


class TestRunPluginIntegration(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerRunPlugin

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"plugin": "youtube_transcript", "params": {"video": "dQw4w9WgXcQ"}}
