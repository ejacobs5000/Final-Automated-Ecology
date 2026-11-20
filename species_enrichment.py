"""
Real per-species data from Wikidata: IUCN Red List conservation status and
taxonomic family — no API key needed for either.

IUCN's own API requires a personal/institutional token obtained by signing
up on their site — not something this app can obtain on a user's behalf.
Wikidata carries conservation status as a structured property (P141,
"IUCN conservation status") sourced from the IUCN Red List, and lets you
walk a species' parent-taxon chain (P171) up to whichever ancestor is
ranked "family" (P105 = Q35409) to get its taxonomic family (as a formal
Latin name via P225, not Wikidata's label — labels are inconsistently
Latin vs. English per family, e.g. "Strigidae" vs. "true owls"). Both are
queried through Wikidata's public SPARQL endpoint with no key and no
signup, matched by exact scientific name (P225, "taxon name").

This is a convenience lookup, not a replacement for IUCN's own authority —
if you need an authoritative, licensable feed (e.g. for publication),
IUCN's own API is still the right source. For this app, "Not assessed"
(status) or a missing family grouping always means "no match was found
here," never a claim about the species itself.
"""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
import json

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "EcologyReportingApp/1.0 (student project; contact via GitHub repo)"
TIMEOUT_SECONDS = 8

_cache: dict[str, dict] = {}  # scientific_name -> {"status": str|None, "family": str|None}


def _escape(name: str) -> str:
    return name.replace("\\", "\\\\").replace('"', '\\"')


def _query_wikidata(names: list[str]) -> dict[str, dict]:
    values = " ".join(f'"{_escape(n)}"' for n in names)
    query = f"""
    SELECT ?taxonName ?statusLabel ?familyName WHERE {{
      VALUES ?taxonName {{ {values} }}
      ?taxon wdt:P225 ?taxonName.
      OPTIONAL {{ ?taxon wdt:P141 ?status. }}
      OPTIONAL {{
        ?taxon wdt:P171+ ?family.
        ?family wdt:P105 wd:Q35409.
        ?family wdt:P225 ?familyName.
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    url = SPARQL_ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    request = urllib.request.Request(
        url, headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        payload = json.load(response)

    results: dict[str, dict] = {}
    for row in payload["results"]["bindings"]:
        name = row["taxonName"]["value"]
        entry = results.setdefault(name, {})
        if "statusLabel" in row:
            entry["status"] = row["statusLabel"]["value"].title()
        if "familyName" in row:
            entry["family"] = row["familyName"]["value"]
    return results


def _lookup(names: list[str]) -> dict[str, dict]:
    """{scientific_name: {"status": ..., "family": ...}}. Cached across
    calls; never raises — a network failure just means nothing new is
    found this time, and callers keep whatever placeholder they had."""
    to_fetch = [n for n in set(names) if n not in _cache]
    if to_fetch:
        try:
            found = _query_wikidata(to_fetch)
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
            print(f"Wikidata species lookup failed: {exc}", flush=True)
            found = {}
        for name in to_fetch:
            _cache[name] = found.get(name, {})

    return {name: _cache[name] for name in names}


def enrich_species(entries: list[dict]) -> None:
    """Fill in entry["status"] (if still "Not assessed") and
    entry["family_name"] (if not already set) from Wikidata. Mutates
    entries in place; skips the lookup entirely if nothing needs it."""
    needing = [
        e for e in entries
        if e.get("status") == "Not assessed" or not e.get("family_name")
    ]
    if not needing:
        return

    info = _lookup([e["scientific_name"] for e in needing])
    for entry in needing:
        data = info.get(entry["scientific_name"], {})
        if entry.get("status") == "Not assessed" and data.get("status"):
            entry["status"] = data["status"]
        if not entry.get("family_name") and data.get("family"):
            entry["family_name"] = data["family"]
