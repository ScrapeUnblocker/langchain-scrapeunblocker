"""LangChain tools for ScrapeUnblocker: read pages, search Google, run site plugins.

Every call goes through the official ``scrapeunblocker`` SDK. The reading tools
mirror the ScrapeUnblocker MCP server's ``fetch_markdown`` and
``search_and_read`` (same caps and output layout), so an agent behaves the same
whichever way it reaches the API.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from pydantic import BaseModel, Field
from scrapeunblocker.exceptions import ScrapeUnblockerError

from langchain_scrapeunblocker import _catalog
from langchain_scrapeunblocker._base import ScrapeUnblockerBaseTool, cap_text, error_text

PAGE_MAX_CHARS = 100_000
SEARCH_PAGE_MAX_CHARS = 15_000
SEARCH_DEFAULT_RESULTS = 3
SEARCH_MAX_RESULTS = 5
SEARCH_PAGE_DEADLINE_S = 120.0
JSON_MAX_CHARS = 60_000

_COUNTRY = Field(
    default=None,
    min_length=2,
    max_length=2,
    description="Optional ISO country code to make the request from, e.g. 'US' or 'DE'.",
)


def _page_text(url: str, markdown: str, max_chars: int) -> str:
    return cap_text(f"URL: {url}\n\n{markdown.strip()}", max_chars, "Ask for a narrower page if you need the rest.")


def _json_text(value: Any, max_chars: int) -> str:
    return cap_text(json.dumps(value, ensure_ascii=False, default=str), max_chars, "The full result is in the artifact.")


# --------------------------------------------------------------------------- markdown


class MarkdownInput(BaseModel):
    url: str = Field(description="The absolute URL to read (http or https).")
    main_content_only: bool = Field(
        default=True,
        description="Drop navigation, page header, footer and sidebars (default). False keeps the whole page.",
    )
    proxy_country: Optional[str] = _COUNTRY


class ScrapeUnblockerMarkdown(ScrapeUnblockerBaseTool):
    """Read a web page as clean Markdown through ScrapeUnblocker.

    Setup:
        .. code-block:: bash

            pip install -U langchain-scrapeunblocker
            export SCRAPEUNBLOCKER_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_scrapeunblocker import ScrapeUnblockerMarkdown

            tool = ScrapeUnblockerMarkdown()

    Invoke:
        .. code-block:: python

            tool.invoke({"url": "https://example.com"})
    """

    name: str = "scrapeunblocker_markdown"
    description: str = (
        "Read a web page in a real browser and return it as clean Markdown: the main content "
        "with absolute links, without scripts, styles, navigation or footer. Use it to read, "
        "quote or summarise a page. Works on JavaScript-rendered pages and on pages that refuse "
        "ordinary fetches (403/429, captcha, Cloudflare). Billed as one request."
    )
    args_schema: Type[BaseModel] = MarkdownInput
    max_chars: int = PAGE_MAX_CHARS
    """Longer pages are cut here, with a note saying so."""

    def _run(
        self,
        url: str,
        main_content_only: bool = True,
        proxy_country: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            md = self.client.get_markdown(url, main_content_only=main_content_only, proxy_country=proxy_country)
        except ScrapeUnblockerError as err:
            raise self._fail(err) from err
        return _page_text(url, md, self.max_chars)

    async def _arun(
        self,
        url: str,
        main_content_only: bool = True,
        proxy_country: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        try:
            md = await self.aclient.get_markdown(url, main_content_only=main_content_only, proxy_country=proxy_country)
        except ScrapeUnblockerError as err:
            raise self._fail(err) from err
        return _page_text(url, md, self.max_chars)


# --------------------------------------------------------------------------- parsed data


class ParsedInput(BaseModel):
    url: str = Field(description="The absolute URL to extract data from (product, article, listing...).")
    proxy_country: Optional[str] = _COUNTRY


class ScrapeUnblockerParsedData(ScrapeUnblockerBaseTool):
    """Extract structured JSON (product, article, listing fields) from a page.

    The content the model sees is the extracted ``data`` as JSON; the artifact is
    the full API response.

    Instantiate:
        .. code-block:: python

            from langchain_scrapeunblocker import ScrapeUnblockerParsedData

            tool = ScrapeUnblockerParsedData()
    """

    name: str = "scrapeunblocker_parsed_data"
    description: str = (
        "Fetch a web page and return its structured data as JSON (for example a product's name, "
        "price, currency, availability and images, or an article's title, author and date) "
        "instead of raw page text. Use it when you need specific fields from a product, article "
        "or listing page. Billed as one request."
    )
    args_schema: Type[BaseModel] = ParsedInput
    response_format: str = "content_and_artifact"
    max_chars: int = JSON_MAX_CHARS

    def _result(self, url: str, page: Any) -> Tuple[str, Dict[str, Any]]:
        if not page.data_extracted:
            return (
                f"URL: {url}\n\nNo structured data could be extracted from this page "
                f"({page.detail or 'nothing recognisable'}). Billed as one request. "
                "Use scrapeunblocker_markdown to read the page instead.",
                page.raw,
            )
        content = {"url": url, "page_type": page.page_type, "source": page.source, "data": page.data}
        return _json_text(content, self.max_chars), page.raw

    def _run(
        self,
        url: str,
        proxy_country: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            page = self.client.get_parsed(url, proxy_country=proxy_country)
        except ScrapeUnblockerError as err:
            raise self._fail(err) from err
        return self._result(url, page)

    async def _arun(
        self,
        url: str,
        proxy_country: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            page = await self.aclient.get_parsed(url, proxy_country=proxy_country)
        except ScrapeUnblockerError as err:
            raise self._fail(err) from err
        return self._result(url, page)


# --------------------------------------------------------------------------- Google search


def _organic(serp: Any) -> List[Dict[str, str]]:
    items = serp.get("organic") if isinstance(serp, dict) else None
    out: List[Dict[str, str]] = []
    seen = set()
    for item in items if isinstance(items, list) else []:
        url = item.get("url") if isinstance(item, dict) else None
        if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "title": str(item.get("title") or url),
                "url": url,
                "description": str(item.get("description") or ""),
            }
        )
    return out


def _ai_overview(serp: Any) -> Optional[str]:
    overview = serp.get("aiOverview") if isinstance(serp, dict) else None
    if not isinstance(overview, dict) or not str(overview.get("text") or "").strip():
        return None
    text = str(overview["text"]).strip()
    sources = [s for s in overview.get("sources") or [] if isinstance(s, str)]
    return f"{text}\n\nSources: {', '.join(sources)}" if sources else text


def _strings(serp: Any, key: str, field: str) -> List[str]:
    items = serp.get(key) if isinstance(serp, dict) else None
    out = []
    for item in items if isinstance(items, list) else []:
        value = item.get(field) if isinstance(item, dict) else item
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return out


def _serp_text(query: str, serp: Any) -> str:
    organic = _organic(serp)
    overview = _ai_overview(serp)
    if not organic and not overview:
        return f'Google returned no results for "{query}". The search was billed as one request.'
    parts = [f'Google results for "{query}":']
    if overview:
        parts.append(f"## AI Overview\n\n{overview}")
    if organic:
        parts.append(
            "## Results\n\n"
            + "\n".join(f"{i}. [{r['title']}]({r['url']})\n   {r['description']}" for i, r in enumerate(organic, 1))
        )
    questions = _strings(serp, "peopleAlsoAsk", "question")
    if questions:
        parts.append("## People also ask\n\n" + "\n".join(f"- {q}" for q in questions))
    related = _strings(serp, "relatedSearches", "query")
    if related:
        parts.append("## Related searches\n\n" + "\n".join(f"- {q}" for q in related))
    return "\n\n".join(parts)


class GoogleSearchInput(BaseModel):
    query: str = Field(description="The search query.")
    proxy_country: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Optional ISO country code to search from, e.g. 'US'. Google localises results to it.",
    )
    pages: int = Field(default=1, ge=1, le=5, description="How many result pages to fetch (1-5, default 1).")


class ScrapeUnblockerGoogleSearch(ScrapeUnblockerBaseTool):
    """Google search results (organic, AI Overview, People also ask) via ScrapeUnblocker.

    The content the model sees is a compact Markdown list; the artifact is the
    full SERP JSON (ads, knowledge panel, local, shopping, news, videos...).

    Instantiate:
        .. code-block:: python

            from langchain_scrapeunblocker import ScrapeUnblockerGoogleSearch

            tool = ScrapeUnblockerGoogleSearch()
    """

    name: str = "scrapeunblocker_google_search"
    description: str = (
        "Search Google and get the results: titles, real URLs and snippets of the organic "
        "results, Google's AI Overview when there is one, People also ask questions and related "
        "searches. Use it to find pages or check what ranks for a query; follow up with "
        "scrapeunblocker_markdown to read a result. Billed as one request per result page."
    )
    args_schema: Type[BaseModel] = GoogleSearchInput
    response_format: str = "content_and_artifact"

    def _run(
        self,
        query: str,
        proxy_country: Optional[str] = None,
        pages: int = 1,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Any]:
        try:
            serp = self.client.serp(query, proxy_country=proxy_country, pages_to_check=pages)
        except ScrapeUnblockerError as err:
            raise self._fail(err) from err
        return _serp_text(query, serp), serp

    async def _arun(
        self,
        query: str,
        proxy_country: Optional[str] = None,
        pages: int = 1,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Any]:
        try:
            serp = await self.aclient.serp(query, proxy_country=proxy_country, pages_to_check=pages)
        except ScrapeUnblockerError as err:
            raise self._fail(err) from err
        return _serp_text(query, serp), serp


# --------------------------------------------------------------------------- search and read


class SearchAndReadInput(BaseModel):
    query: str = Field(description="The search query.")
    results: int = Field(
        default=SEARCH_DEFAULT_RESULTS,
        ge=1,
        le=SEARCH_MAX_RESULTS,
        description=f"How many top results to read (default {SEARCH_DEFAULT_RESULTS}, max {SEARCH_MAX_RESULTS}).",
    )
    proxy_country: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Optional ISO country code to search from and read the pages through, e.g. 'US'.",
    )


def _search_and_read_text(
    query: str, overview: Optional[str], to_read: List[Dict[str, str]], pages: List[Any], rest: List[Dict[str, str]]
) -> str:
    parts = [f'Google results for "{query}": {len(to_read)} read below. Billed as one search plus one request per page read.']
    if overview:
        parts.append(f"## Google AI Overview\n\n{overview}")
    for i, (r, page) in enumerate(zip(to_read, pages), 1):
        if isinstance(page, BaseException):
            reason = error_text(page) if isinstance(page, ScrapeUnblockerError) else str(page) or type(page).__name__
            body = f"URL: {r['url']}\n\nCould not read this page: {reason}\nSnippet: {r['description']}"
        else:
            body = _page_text(r["url"], page, SEARCH_PAGE_MAX_CHARS)
        parts.append(f"## {i}. {r['title']}\n\n{body}")
    if rest:
        parts.append("## Other results\n\n" + "\n".join(f"- [{r['title']}]({r['url']}): {r['description']}" for r in rest))
    return "\n\n".join(parts)


class ScrapeUnblockerSearchAndRead(ScrapeUnblockerBaseTool):
    """Search Google and read the top results as Markdown, in one tool call.

    Instantiate:
        .. code-block:: python

            from langchain_scrapeunblocker import ScrapeUnblockerSearchAndRead

            tool = ScrapeUnblockerSearchAndRead()
    """

    name: str = "scrapeunblocker_search_and_read"
    description: str = (
        "Search Google and read the top results in one call: returns Google's AI Overview when "
        f"there is one, each of the first `results` pages (default {SEARCH_DEFAULT_RESULTS}, max "
        f"{SEARCH_MAX_RESULTS}) as Markdown, and the other results as title, URL and snippet. Use "
        "it to answer a question from the live web. Billed as one request for the search plus "
        "one per page read."
    )
    args_schema: Type[BaseModel] = SearchAndReadInput

    def _run(
        self,
        query: str,
        results: int = SEARCH_DEFAULT_RESULTS,
        proxy_country: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            serp = self.client.serp(query, proxy_country=proxy_country)
        except ScrapeUnblockerError as err:
            raise self._fail(err) from err
        organic, overview = _organic(serp), _ai_overview(serp)
        if not organic and not overview:
            return f'Google returned no organic results for "{query}". The search was billed as one request.'
        to_read = organic[:results]
        pages: List[Any] = []
        deadline = time.monotonic() + SEARCH_PAGE_DEADLINE_S
        pool = ThreadPoolExecutor(max_workers=max(1, len(to_read)))
        try:
            futures = [
                pool.submit(self.client.get_markdown, r["url"], proxy_country=proxy_country) for r in to_read
            ]
            for future in futures:
                try:
                    pages.append(future.result(timeout=max(0.0, deadline - time.monotonic())))
                except FutureTimeout:
                    pages.append(TimeoutError(f"no answer within {SEARCH_PAGE_DEADLINE_S:.0f} s"))
                except Exception as err:  # noqa: BLE001 - one unreadable page must not fail the rest
                    pages.append(err)
        finally:
            # A page past the deadline is not waited for.
            pool.shutdown(wait=False, cancel_futures=True)
        return _search_and_read_text(query, overview, to_read, pages, organic[len(to_read):])

    async def _arun(
        self,
        query: str,
        results: int = SEARCH_DEFAULT_RESULTS,
        proxy_country: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        try:
            serp = await self.aclient.serp(query, proxy_country=proxy_country)
        except ScrapeUnblockerError as err:
            raise self._fail(err) from err
        organic, overview = _organic(serp), _ai_overview(serp)
        if not organic and not overview:
            return f'Google returned no organic results for "{query}". The search was billed as one request.'
        to_read = organic[:results]
        pages = await asyncio.gather(
            *(
                asyncio.wait_for(
                    self.aclient.get_markdown(r["url"], proxy_country=proxy_country), SEARCH_PAGE_DEADLINE_S
                )
                for r in to_read
            ),
            return_exceptions=True,
        )
        return _search_and_read_text(query, overview, to_read, list(pages), organic[len(to_read):])


# --------------------------------------------------------------------------- plugins


class ListPluginsInput(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description=(
            "Words to look for in plugin names and descriptions, e.g. 'amazon', 'reviews', 'jobs'. "
            "Pass a plugin's exact name to get its full parameter documentation. Leave empty to list all."
        ),
    )


class ScrapeUnblockerListPlugins(ScrapeUnblockerBaseTool):
    """List ScrapeUnblocker's ready-made site plugins (100+ sites, structured JSON).

    Makes no API call: the catalog comes with the installed ``scrapeunblocker`` SDK.
    """

    name: str = "scrapeunblocker_list_plugins"
    description: str = (
        "List ScrapeUnblocker's ready-made site scrapers (plugins) that return structured JSON "
        "for 100+ sites: Amazon, Walmart, eBay, Google Maps / Local / Shopping / Images, YouTube, "
        "Zillow, Indeed, Trustpilot, Booking.com, TikTok and more. Search by keyword, or pass a "
        "plugin's exact name to see its parameters, then call it with scrapeunblocker_run_plugin. "
        "Free: makes no request."
    )
    args_schema: Type[BaseModel] = ListPluginsInput

    def _run(self, query: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return _catalog.describe(query)

    async def _arun(
        self, query: Optional[str] = None, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return _catalog.describe(query)


class RunPluginInput(BaseModel):
    plugin: str = Field(description="The plugin's name as listed by scrapeunblocker_list_plugins, e.g. 'amazon_product'.")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="The plugin's parameters by name, e.g. {\"asin\": \"B08N5WRWNW\", \"marketplace\": \"com\"}.",
    )


class ScrapeUnblockerRunPlugin(ScrapeUnblockerBaseTool):
    """Run one of ScrapeUnblocker's site plugins and get its structured JSON.

    The content the model sees is the JSON (cut at ``max_chars``); the artifact
    is the full response.
    """

    name: str = "scrapeunblocker_run_plugin"
    description: str = (
        "Run a ScrapeUnblocker site plugin and get structured JSON, e.g. an Amazon product, "
        "Walmart or eBay search results, Google Maps places, a YouTube transcript or Zillow "
        "listings. Find the plugin and its parameters with scrapeunblocker_list_plugins first. "
        "Billed per call like other requests (see the plugin's description for exceptions)."
    )
    args_schema: Type[BaseModel] = RunPluginInput
    response_format: str = "content_and_artifact"
    max_chars: int = JSON_MAX_CHARS

    def _run(
        self,
        plugin: str,
        params: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Any]:
        kwargs = _catalog.check_call(plugin, params or {})
        try:
            result = getattr(self.client, plugin)(**kwargs)
        except ScrapeUnblockerError as err:
            raise self._fail(err) from err
        return _json_text(result, self.max_chars), result

    async def _arun(
        self,
        plugin: str,
        params: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Any]:
        kwargs = _catalog.check_call(plugin, params or {})
        try:
            result = await getattr(self.aclient, plugin)(**kwargs)
        except ScrapeUnblockerError as err:
            raise self._fail(err) from err
        return _json_text(result, self.max_chars), result


__all__ = [
    "ScrapeUnblockerMarkdown",
    "ScrapeUnblockerParsedData",
    "ScrapeUnblockerGoogleSearch",
    "ScrapeUnblockerSearchAndRead",
    "ScrapeUnblockerListPlugins",
    "ScrapeUnblockerRunPlugin",
]
