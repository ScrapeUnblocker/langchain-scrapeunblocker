# Changelog

## 0.2.0

- `ScrapeUnblockerGoogleSearch` takes an optional `full_urls` (default `False`): wait until Google gives out every result's real URL. Google releases a results page's links a few at a time, the last ones up to about 40 seconds after the search. Without it the search returns sooner, and a result whose link has not come yet has the site's home page as its `url` (or a Google redirect link when Google shows no site). With it, the search takes up to about 40 seconds longer. Billed the same. The model is told to set it when it needs the exact URL of every result, e.g. to cite or open them.
- Needs `scrapeunblocker>=0.19.0`, the SDK version whose `serp()` takes `full_urls`.

## 0.1.0

First release: `ScrapeUnblockerMarkdown`, `ScrapeUnblockerSearchAndRead`, `ScrapeUnblockerGoogleSearch`, `ScrapeUnblockerParsedData`, `ScrapeUnblockerListPlugins`, `ScrapeUnblockerRunPlugin` and `ScrapeUnblockerToolkit`, sync and async, passing LangChain's standard tool tests.
