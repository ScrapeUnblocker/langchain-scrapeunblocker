"""The site plugin catalog, read from the installed ``scrapeunblocker`` SDK.

The SDK has one generated method per public plugin, with its parameters and a
docstring. Reading them here keeps this package in step with the SDK: a new
plugin needs an SDK upgrade, not a release of this package.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional

from langchain_core.tools import ToolException
from scrapeunblocker import Client

# Plugins the SDK still has methods for but we do not offer.
_HIDDEN_PREFIXES = ("temu_",)
# Client methods that are not plugins.
_NOT_PLUGINS = frozenset(
    {
        "close",
        "plugin",
        "get_page_source",
        "get_markdown",
        "get_parsed",
        "get_page_with_cookies",
        "get_image",
        "serp",
    }
)
_SUMMARY_MAX = 110
_LIST_MAX = 40


class Plugin(NamedTuple):
    name: str
    summary: str
    doc: str
    params: List[str]
    signature: inspect.Signature


def _summary(doc: str) -> str:
    first = " ".join(doc.split("\n\n", 1)[0].split())
    return first if len(first) <= _SUMMARY_MAX else first[: _SUMMARY_MAX - 3].rstrip() + "..."


@lru_cache(maxsize=1)
def plugins() -> Dict[str, Plugin]:
    found: Dict[str, Plugin] = {}
    for name, member in inspect.getmembers(Client, inspect.isfunction):
        if name.startswith("_") or name in _NOT_PLUGINS or name.startswith(_HIDDEN_PREFIXES):
            continue
        sig = inspect.signature(member)
        params = [p for p in sig.parameters if p != "self" and sig.parameters[p].kind != inspect.Parameter.VAR_KEYWORD]
        doc = inspect.getdoc(member) or ""
        found[name] = Plugin(name, _summary(doc), doc, params, sig)
    return found


def _line(p: Plugin) -> str:
    return f"- {p.name}({', '.join(p.params)}): {p.summary}"


def describe(query: Optional[str] = None) -> str:
    """The catalog text for scrapeunblocker_list_plugins."""
    catalog = plugins()
    q = (query or "").strip().lower()
    if q in catalog:
        p = catalog[q]
        return f"{p.name}({', '.join(p.params)})\n\n{p.doc}"
    if not q:
        # Names only: with summaries the whole list is too long for one tool result.
        return (
            f"{len(catalog)} plugins. Search with a word (e.g. 'amazon', 'reviews', 'jobs') to see what "
            "they do, or pass an exact name for its parameters.\n\n" + ", ".join(catalog)
        )
    words = q.split()
    matches = [p for p in catalog.values() if all(w in f"{p.name} {p.summary}".lower() for w in words)]
    if not matches:
        return f'No plugin matches "{query}". Call scrapeunblocker_list_plugins with no query to see all of them.'
    shown = matches[:_LIST_MAX]
    more = f" (showing {_LIST_MAX}; narrow the query)" if len(matches) > _LIST_MAX else ""
    return f'{len(matches)} plugins match "{query}"{more}:\n\n' + "\n".join(_line(p) for p in shown)


def check_call(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """The keyword arguments for plugin `name`, or a ToolException the agent can fix."""
    p = plugins().get(name)
    if p is None:
        raise ToolException(
            f'There is no plugin "{name}". Find the right name with scrapeunblocker_list_plugins.'
        )
    if not isinstance(params, dict):
        raise ToolException("params must be an object of parameter names to values.")
    unknown = sorted(set(params) - set(p.params))
    if unknown:
        raise ToolException(
            f"{name} has no parameter {', '.join(unknown)}. Its parameters: {', '.join(p.params) or 'none'}."
        )
    try:
        p.signature.bind(None, **params)
    except TypeError as err:
        raise ToolException(f"Wrong parameters for {name}: {err}. Its parameters: {', '.join(p.params)}.") from err
    return params
