"""LangChain's standard unit tests for every tool (no network)."""

from typing import Any, Type

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_scrapeunblocker import (
    ScrapeUnblockerGoogleSearch,
    ScrapeUnblockerListPlugins,
    ScrapeUnblockerMarkdown,
    ScrapeUnblockerParsedData,
    ScrapeUnblockerRunPlugin,
    ScrapeUnblockerSearchAndRead,
)


class _Base(ToolsUnitTests):
    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-key"}

    @property
    def init_from_env_params(self) -> tuple:
        return {"SCRAPEUNBLOCKER_API_KEY": "env-key"}, {}, {"api_key": "env-key"}


class TestMarkdownUnit(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerMarkdown

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": "https://example.com"}


class TestParsedDataUnit(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerParsedData

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": "https://example.com/product"}


class TestGoogleSearchUnit(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerGoogleSearch

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "best web scraping api"}


class TestSearchAndReadUnit(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerSearchAndRead

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "langchain tools", "results": 2}


class TestListPluginsUnit(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerListPlugins

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "amazon"}


class TestRunPluginUnit(_Base):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return ScrapeUnblockerRunPlugin

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"plugin": "amazon_product", "params": {"asin": "B08N5WRWNW"}}
