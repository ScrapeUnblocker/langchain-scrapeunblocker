"""The tools against a fake SDK client: what the model sees, errors, async."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from langchain_core.messages import ToolMessage
from scrapeunblocker.exceptions import AuthenticationError, BlockedError, TargetNotFoundError
from scrapeunblocker.models import ParsedPage

from langchain_scrapeunblocker import (
    ScrapeUnblockerGoogleSearch,
    ScrapeUnblockerListPlugins,
    ScrapeUnblockerMarkdown,
    ScrapeUnblockerParsedData,
    ScrapeUnblockerRunPlugin,
    ScrapeUnblockerSearchAndRead,
    ScrapeUnblockerToolkit,
)

PAGES = {
    "https://shop.example/item": "# Item\n\nPrice: $19.99",
    "https://shop.example/long": "x" * 150_000,
}
SERP = {
    "organic": [
        {"title": "Item", "url": "https://shop.example/item", "description": "first"},
        {"title": "Gone", "url": "https://shop.example/gone", "description": "second"},
        {"title": "Other", "url": "https://other.example/", "description": "third"},
        {"title": "Dup", "url": "https://shop.example/item", "description": "duplicate"},
    ],
    "aiOverview": {"text": "Items cost $19.99.", "sources": ["https://shop.example/item"]},
    "peopleAlsoAsk": [{"question": "Is it cheap?"}],
    "relatedSearches": [{"query": "item price"}],
}


def _not_found(url: str) -> TargetNotFoundError:
    return TargetNotFoundError("gone", status_code=404, origin_status=404, html="# Not found")


class FakeClient:
    """Stands in for scrapeunblocker.Client and AsyncClient (same method names)."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def get_markdown(self, url: str, **kw: Any) -> str:
        self.calls.append(("get_markdown", url, kw))
        if url in PAGES:
            return PAGES[url]
        raise _not_found(url)

    def get_parsed(self, url: str, **kw: Any) -> ParsedPage:
        self.calls.append(("get_parsed", url, kw))
        if url.endswith("/empty"):
            return ParsedPage.from_response({"data_extracted": False, "detail": "no schema", "html": "<p>x</p>"})
        return ParsedPage.from_response({"data": {"page_type": "product", "source": "schema.org", "data": {"price": 19.99}}})

    def serp(self, keyword: str, **kw: Any) -> Dict[str, Any]:
        self.calls.append(("serp", keyword, kw))
        if keyword == "nothing":
            return {"organic": []}
        return SERP

    def amazon_product(self, **kw: Any) -> Dict[str, Any]:
        self.calls.append(("amazon_product", kw))
        return {"asin": kw.get("asin"), "title": "Echo Dot"}


class AsyncFakeClient(FakeClient):
    async def get_markdown(self, url: str, **kw: Any) -> str:  # type: ignore[override]
        return FakeClient.get_markdown(self, url, **kw)

    async def get_parsed(self, url: str, **kw: Any) -> ParsedPage:  # type: ignore[override]
        return FakeClient.get_parsed(self, url, **kw)

    async def serp(self, keyword: str, **kw: Any) -> Dict[str, Any]:  # type: ignore[override]
        return FakeClient.serp(self, keyword, **kw)

    async def amazon_product(self, **kw: Any) -> Dict[str, Any]:  # type: ignore[override]
        return FakeClient.amazon_product(self, **kw)


def make(cls: type, **kw: Any) -> Any:
    tool = cls(api_key="test-key", **kw)
    tool._client = FakeClient()
    tool._aclient = AsyncFakeClient()
    return tool


def call(tool: Any, args: Dict[str, Any]) -> ToolMessage:
    return tool.invoke({"name": tool.name, "args": args, "id": "1", "type": "tool_call"})


# ------------------------------------------------------------------ markdown


def test_markdown_asks_for_main_content_and_labels_the_url() -> None:
    tool = make(ScrapeUnblockerMarkdown)
    out = tool.invoke({"url": "https://shop.example/item"})
    assert out == "URL: https://shop.example/item\n\n# Item\n\nPrice: $19.99"
    assert tool._client.calls == [
        ("get_markdown", "https://shop.example/item", {"main_content_only": True, "proxy_country": None})
    ]


def test_markdown_cuts_a_long_page_and_says_so() -> None:
    out = make(ScrapeUnblockerMarkdown).invoke({"url": "https://shop.example/long"})
    assert len(out) < 101_000
    assert "[Cut at 100000 of" in out


def test_markdown_target_404_comes_back_as_text() -> None:
    msg = call(make(ScrapeUnblockerMarkdown), {"url": "https://shop.example/missing"})
    assert "answered 404" in msg.content
    assert "do not retry" in msg.content


@pytest.mark.parametrize(
    "err, words",
    [
        (AuthenticationError("Unauthorized", status_code=401), "rejected the API key"),
        (BlockedError("blocked", status_code=403), "Not billed"),
    ],
)
def test_api_errors_tell_the_agent_what_happened(err: Exception, words: str) -> None:
    tool = make(ScrapeUnblockerMarkdown)

    def boom(url: str, **kw: Any) -> str:
        raise err

    tool._client.get_markdown = boom
    assert words in call(tool, {"url": "https://shop.example/item"}).content


def test_missing_key_is_reported_on_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCRAPEUNBLOCKER_API_KEY", raising=False)
    monkeypatch.delenv("SCRAPEUNBLOCKER_KEY", raising=False)
    tool = ScrapeUnblockerMarkdown()
    assert "No ScrapeUnblocker API key" in call(tool, {"url": "https://example.com"}).content


def test_key_falls_back_to_the_sdk_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCRAPEUNBLOCKER_API_KEY", raising=False)
    monkeypatch.setenv("SCRAPEUNBLOCKER_KEY", "sdk-key")
    assert ScrapeUnblockerMarkdown().api_key.get_secret_value() == "sdk-key"


def test_user_agent_names_this_package() -> None:
    tool = ScrapeUnblockerMarkdown(api_key="test-key")
    assert "langchain-scrapeunblocker/" in tool.client._http.headers["User-Agent"]
    assert "scrapeunblocker-python/" in tool.client._http.headers["User-Agent"]


async def test_markdown_async() -> None:
    out = await make(ScrapeUnblockerMarkdown).ainvoke({"url": "https://shop.example/item"})
    assert out.startswith("URL: https://shop.example/item")


# ------------------------------------------------------------------ parsed data


def test_parsed_content_is_the_data_artifact_the_response() -> None:
    msg = call(make(ScrapeUnblockerParsedData), {"url": "https://shop.example/item"})
    assert '"page_type": "product"' in msg.content
    assert '"price": 19.99' in msg.content
    assert msg.artifact["data"]["data"] == {"price": 19.99}


def test_parsed_nothing_extracted_points_to_markdown() -> None:
    msg = call(make(ScrapeUnblockerParsedData), {"url": "https://shop.example/empty"})
    assert "No structured data" in msg.content
    assert "scrapeunblocker_markdown" in msg.content


# ------------------------------------------------------------------ Google search


def test_google_search_lists_results_overview_and_extras() -> None:
    msg = call(make(ScrapeUnblockerGoogleSearch), {"query": "item", "proxy_country": "US"})
    assert "## AI Overview\n\nItems cost $19.99." in msg.content
    assert "1. [Item](https://shop.example/item)" in msg.content
    assert "Dup" not in msg.content  # same URL twice is listed once
    assert "- Is it cheap?" in msg.content and "- item price" in msg.content
    assert msg.artifact is SERP


def test_google_search_waits_for_full_urls_only_when_asked() -> None:
    tool = make(ScrapeUnblockerGoogleSearch)
    call(tool, {"query": "item"})
    call(tool, {"query": "item", "full_urls": True})
    assert [c[2]["full_urls"] for c in tool._client.calls] == [False, True]
    assert "full_urls" in tool.description


async def test_google_search_full_urls_async() -> None:
    tool = make(ScrapeUnblockerGoogleSearch)
    msg = await tool.ainvoke(
        {"name": tool.name, "args": {"query": "item", "full_urls": True}, "id": "1", "type": "tool_call"}
    )
    assert "1. [Item](https://shop.example/item)" in msg.content
    assert tool._aclient.calls[-1][2]["full_urls"] is True


def test_google_search_no_results() -> None:
    msg = call(make(ScrapeUnblockerGoogleSearch), {"query": "nothing"})
    assert "no results" in msg.content


# ------------------------------------------------------------------ search and read


def test_search_and_read_reads_top_results_and_survives_a_bad_page() -> None:
    tool = make(ScrapeUnblockerSearchAndRead)
    out = tool.invoke({"query": "item", "results": 2})
    assert out.startswith('Google results for "item": 2 read below.')
    assert "## 1. Item\n\nURL: https://shop.example/item\n\n# Item" in out
    assert "## 2. Gone\n\nURL: https://shop.example/gone\n\nCould not read this page: The page itself answered 404" in out
    assert "## Other results\n\n- [Other](https://other.example/): third" in out
    read = [c[1] for c in tool._client.calls if c[0] == "get_markdown"]
    assert sorted(read) == ["https://shop.example/gone", "https://shop.example/item"]


async def test_search_and_read_async() -> None:
    out = await make(ScrapeUnblockerSearchAndRead).ainvoke({"query": "item", "results": 1})
    assert "## 1. Item" in out and "## Google AI Overview" in out


def test_search_and_read_caps_results_at_five() -> None:
    with pytest.raises(Exception):
        make(ScrapeUnblockerSearchAndRead).invoke({"query": "item", "results": 9})


# ------------------------------------------------------------------ plugins


def test_list_plugins_search_and_detail() -> None:
    tool = make(ScrapeUnblockerListPlugins)
    found = tool.invoke({"query": "amazon product"})
    assert "amazon_product(" in found
    detail = tool.invoke({"query": "amazon_product"})
    assert detail.startswith("amazon_product(") and "Args:" in detail
    assert "temu" not in tool.invoke({}).lower()
    assert tool._client.calls == []  # free: no API call


def test_run_plugin_calls_the_sdk_method() -> None:
    tool = make(ScrapeUnblockerRunPlugin)
    msg = call(tool, {"plugin": "amazon_product", "params": {"asin": "B08N5WRWNW"}})
    assert '"title": "Echo Dot"' in msg.content
    assert msg.artifact == {"asin": "B08N5WRWNW", "title": "Echo Dot"}
    assert tool._client.calls == [("amazon_product", {"asin": "B08N5WRWNW"})]


@pytest.mark.parametrize(
    "args, words",
    [
        ({"plugin": "no_such_plugin", "params": {}}, "There is no plugin"),
        ({"plugin": "amazon_product", "params": {"color": "red"}}, "has no parameter color"),
        ({"plugin": "temu_search", "params": {}}, "There is no plugin"),
        ({"plugin": "get_page_source", "params": {}}, "There is no plugin"),
    ],
)
def test_run_plugin_rejects_what_it_cannot_run(args: Dict[str, Any], words: str) -> None:
    tool = make(ScrapeUnblockerRunPlugin)
    assert words in call(tool, args).content
    assert tool._client.calls == []


async def test_run_plugin_async() -> None:
    tool = make(ScrapeUnblockerRunPlugin)
    msg = await tool.ainvoke(
        {"name": tool.name, "args": {"plugin": "amazon_product", "params": {"asin": "X"}}, "id": "1", "type": "tool_call"}
    )
    assert msg.artifact["asin"] == "X"


# ------------------------------------------------------------------ toolkit


def test_toolkit_shares_settings() -> None:
    tools = ScrapeUnblockerToolkit(api_key="tk", base_url="https://api-staging.example").get_tools()
    assert [t.name for t in tools] == [
        "scrapeunblocker_markdown",
        "scrapeunblocker_search_and_read",
        "scrapeunblocker_google_search",
        "scrapeunblocker_parsed_data",
        "scrapeunblocker_list_plugins",
        "scrapeunblocker_run_plugin",
    ]
    assert all(t.api_key.get_secret_value() == "tk" for t in tools)
    assert all(t.base_url == "https://api-staging.example" for t in tools)
    assert len(ScrapeUnblockerToolkit(api_key="tk", include_plugins=False).get_tools()) == 4
