# langchain-scrapeunblocker

[![PyPI](https://img.shields.io/pypi/v/langchain-scrapeunblocker)](https://pypi.org/project/langchain-scrapeunblocker/)
[![Test](https://github.com/ScrapeUnblocker/langchain-scrapeunblocker/actions/workflows/test.yml/badge.svg)](https://github.com/ScrapeUnblocker/langchain-scrapeunblocker/actions/workflows/test.yml)

LangChain tools for [ScrapeUnblocker](https://scrapeunblocker.com?utm_source=github&utm_medium=integration&utm_campaign=langchain): give your agent live web data. Read any public page in a real browser as clean Markdown or structured JSON, search Google, and run 100+ ready-made site scrapers (Amazon, Walmart, eBay, Google Maps, YouTube, Zillow, Indeed and more), including pages behind Cloudflare, DataDome, PerimeterX and Akamai.

## Install

```bash
pip install -U langchain-scrapeunblocker
export SCRAPEUNBLOCKER_API_KEY="your-api-key"
```

Get a key at [app.scrapeunblocker.com](https://app.scrapeunblocker.com?utm_source=github&utm_medium=integration&utm_campaign=langchain) - a free account comes with 500 requests to start.

## Use with an agent

```python
from langchain.agents import create_agent
from langchain_scrapeunblocker import ScrapeUnblockerToolkit

agent = create_agent(
    "anthropic:claude-sonnet-5-5",
    tools=ScrapeUnblockerToolkit().get_tools(),
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What does the Echo Dot cost on amazon.com today?"}]}
)
print(result["messages"][-1].content)
```

## Tools

| Tool | Name the model sees | What it does | Billed |
|------|---------------------|--------------|--------|
| `ScrapeUnblockerMarkdown` | `scrapeunblocker_markdown` | A page as clean Markdown (main content, absolute links) | 1 request |
| `ScrapeUnblockerSearchAndRead` | `scrapeunblocker_search_and_read` | Google search plus the top 1-5 results read as Markdown, in one call | 1 + 1 per page |
| `ScrapeUnblockerGoogleSearch` | `scrapeunblocker_google_search` | Organic results, AI Overview, People also ask, related searches | 1 per result page |
| `ScrapeUnblockerParsedData` | `scrapeunblocker_parsed_data` | Structured JSON from a product, article or listing page | 1 request |
| `ScrapeUnblockerListPlugins` | `scrapeunblocker_list_plugins` | Search the catalog of 100+ site scrapers | free |
| `ScrapeUnblockerRunPlugin` | `scrapeunblocker_run_plugin` | Run a site scraper and get its JSON | per plugin |

Use them one by one:

```python
from langchain_scrapeunblocker import ScrapeUnblockerMarkdown, ScrapeUnblockerRunPlugin

ScrapeUnblockerMarkdown().invoke({"url": "https://example.com"})

ScrapeUnblockerRunPlugin().invoke(
    {"plugin": "youtube_transcript", "params": {"video": "dQw4w9WgXcQ"}}
)
```

`ScrapeUnblockerGoogleSearch`, `ScrapeUnblockerParsedData` and `ScrapeUnblockerRunPlugin` return `content_and_artifact`: the model gets a compact text or JSON, and the full API response is on the `ToolMessage`'s `artifact` for your code. Called with a tool call, they return that message:

```python
from langchain_scrapeunblocker import ScrapeUnblockerGoogleSearch

msg = ScrapeUnblockerGoogleSearch().invoke(
    {"name": "scrapeunblocker_google_search", "args": {"query": "langchain"}, "id": "1", "type": "tool_call"}
)
msg.content    # what the model reads
msg.artifact   # the whole results page: ads, knowledge panel, local, shopping, videos...
```

Every tool takes an optional `proxy_country` (ISO code such as `"US"` or `"DE"`) to make the request from that country, and works with `ainvoke` too.

The plugin catalog comes from the installed [`scrapeunblocker`](https://pypi.org/project/scrapeunblocker/) SDK: `pip install -U scrapeunblocker` picks up new plugins without a new release of this package.

## Configuration

| Argument | Default | |
|----------|---------|---|
| `api_key` | `SCRAPEUNBLOCKER_API_KEY` (or `SCRAPEUNBLOCKER_KEY`) | Your API key |
| `timeout` | `180` | Seconds per API call; protected pages can take a while |
| `max_retries` | `2` | Retries for 429 and 502-504 |
| `max_chars` | `100000` (Markdown), `60000` (JSON) | Longer results are cut, with a note saying so |

`ScrapeUnblockerToolkit(include_plugins=False)` leaves out the two plugin tools.

Failed calls come back to the model as text it can act on (for example "the page itself answered 404, do not retry"), not as exceptions. Set `handle_tool_error=False` on a tool to raise instead.

## Links

- [ScrapeUnblocker docs](https://docs.scrapeunblocker.com?utm_source=github&utm_medium=integration&utm_campaign=langchain)
- [Site plugins](https://docs.scrapeunblocker.com/plugins/overview?utm_source=github&utm_medium=integration&utm_campaign=langchain)
- [Python SDK](https://github.com/ScrapeUnblocker/scrapeunblocker-python)

## Development

```bash
pip install -e ".[test]"
pytest                                 # unit tests, no network
SCRAPEUNBLOCKER_API_KEY=... pytest tests/integration_tests   # real API calls
```

## License

MIT
