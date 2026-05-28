"""Wikipedia page bindings for GeoMosaic Tier 1 events."""

from __future__ import annotations

from urllib.parse import quote


WIKIPEDIA_EVENT_PAGES = {
    "crimea": "Annexation of Crimea by the Russian Federation",
    "iraq": "2003 invasion of Iraq",
    "libya": "2011 military intervention in Libya",
    "kosovo": "2008 Kosovo declaration of independence",
    "scs": "South China Sea Arbitration",
    "jcpoa": "Joint Comprehensive Plan of Action",
    "ukraine": "Russian invasion of Ukraine",
    "hongkong": "2020 Hong Kong national security law",
}

WIKIPEDIA_EVENT_PAGE_ALIASES = {
    "crimea": ("2014 Russian annexation of Crimea",),
    "jcpoa": ("Iran nuclear deal",),
    "ukraine": ("Russo-Ukrainian war (2022–present)",),
}

WIKIPEDIA_EVENT_FALLBACK_PAGES = {
    "hongkong": ("2019–2020 Hong Kong protests",),
    "scs": ("Territorial disputes in the South China Sea",),
}


def wikipedia_page_url(page_title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(page_title.replace(' ', '_'))}"
