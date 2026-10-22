"""Emergency Assistance & Nearby Resources — Geoapify backend.

Turns an operator-entered location (free text) into nearby emergency resources —
police stations, hospitals, clinics, pharmacies and ambulance/emergency
facilities — using the Geoapify Geocoding and Places APIs.

This module is deliberately framework-agnostic (no Streamlit import) so it can be
unit-tested in isolation and reused outside the dashboard. The Streamlit app
supplies the API key (from ``st.secrets`` or the environment) and wraps the
lookups in its own ``st.cache_data`` layer.

SAFETY: results are *informational only*. They are OpenStreetMap-derived listings
surfaced to help a human operator; they do not confirm that a facility is open,
reachable, or appropriate, and they NEVER dispatch emergency services. The name,
phone and hours of a listing may be stale. Always verify before acting.

Configuration:
    GEOAPIFY_API_KEY — required, read from the environment (or passed explicitly).
                       Never hardcoded. Get a free key at https://www.geoapify.com/.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
PLACES_URL = "https://api.geoapify.com/v2/places"

# Google Places (optional) — phone, ratings, review counts, written reviews.
GOOGLE_FIND_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
GOOGLE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
GOOGLE_DETAIL_FIELDS = (
    "name,formatted_phone_number,international_phone_number,rating,"
    "user_ratings_total,reviews,url,geometry"
)
GOOGLE_MATCH_RADIUS_M = 600      # a candidate must be within this of the Geoapify point
NAME_MATCH_MIN = 0.34            # min token-Jaccard name overlap to accept a match
MAX_GOOGLE_ENRICH = 5            # enrich at most N results per type (API-limit friendly)
MAX_REVIEWS = 3                  # written reviews surfaced per resource

DEFAULT_RADIUS_M = 5000      # 5 km search radius
MIN_RADIUS_M = 500
MAX_RADIUS_M = 50000         # Geoapify circle filter caps well above typical use
DEFAULT_LIMIT = 10           # max results per resource type
DEFAULT_TIMEOUT = 10         # seconds per HTTP call

# Friendly resource type -> Geoapify Places category/categories.
# Geoapify's public taxonomy has no dedicated "ambulance station" category, so
# "ambulance" is mapped to hospitals with emergency departments (the practical
# proxy for ambulance dispatch). This is documented, not fabricated — results for
# that type are labelled as emergency/hospital facilities, not literal stations.
RESOURCE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "police": ("service.police",),
    "hospital": ("healthcare.hospital",),
    "clinic": ("healthcare.clinic_or_praxis",),
    "pharmacy": ("healthcare.pharmacy",),
    "ambulance": ("healthcare.hospital",),
}

DEFAULT_RESOURCE_TYPES: tuple[str, ...] = ("police", "hospital", "clinic", "ambulance")

# Shared connection pool.
_SESSION = requests.Session()


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------
class EmergencyServiceError(Exception):
    """Base class for all emergency-lookup failures."""


class MissingApiKeyError(EmergencyServiceError):
    """GEOAPIFY_API_KEY is not configured."""


class LocationNotFoundError(EmergencyServiceError):
    """The entered location could not be geocoded."""


class ApiRequestError(EmergencyServiceError):
    """A Geoapify request timed out, failed, or returned an error status."""


class GooglePlacesError(EmergencyServiceError):
    """A Google Places request failed (denied, over quota/billing, bad request)."""


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Coordinates:
    lat: float
    lon: float
    formatted: str = ""  # Geoapify's canonical address for the matched place


@dataclass
class Resource:
    name: str
    type: str                      # friendly resource type (police/hospital/...)
    lat: float | None = None
    lon: float | None = None
    distance_m: float | None = None
    address: str = ""
    phone: str = ""
    category: str = ""             # raw Geoapify category
    directions_url: str = ""
    # --- optional Google Places enrichment (None/empty when unavailable) ---
    rating: float | None = None
    reviews_count: int | None = None
    reviews: list[dict[str, Any]] = field(default_factory=list)
    google_url: str = ""           # link to the original Google Maps listing
    google_matched: bool = False   # True only when a confident match was found

    @property
    def distance_km(self) -> float | None:
        return None if self.distance_m is None else round(self.distance_m / 1000.0, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "distance_km": self.distance_km,
            "distance_m": None if self.distance_m is None else round(self.distance_m),
            "address": self.address,
            "phone": self.phone,
            "lat": self.lat,
            "lon": self.lon,
            "category": self.category,
            "directions_url": self.directions_url,
            "rating": self.rating,
            "reviews_count": self.reviews_count,
            "reviews": self.reviews,
            "google_url": self.google_url,
            "google_matched": self.google_matched,
        }


@dataclass
class SearchResult:
    query_location: str
    coordinates: Coordinates
    radius_m: int
    resources: dict[str, list[Resource]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)  # per-type failures

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.resources.values())

    @property
    def is_empty(self) -> bool:
        return self.total == 0


# -----------------------------------------------------------------------------
# API key handling
# -----------------------------------------------------------------------------
def get_api_key(explicit: str | None = None) -> str | None:
    """Resolve the Geoapify key: explicit argument first, then the environment.

    The Streamlit app passes a key read from ``st.secrets``; outside the app we
    fall back to the ``GEOAPIFY_API_KEY`` environment variable. The key is never
    hardcoded and never logged.
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    value = os.environ.get("GEOAPIFY_API_KEY")
    return value.strip() if value and value.strip() else None


def _require_key(api_key: str | None) -> str:
    key = get_api_key(api_key)
    if not key:
        raise MissingApiKeyError(
            "GEOAPIFY_API_KEY is not configured. Add it to Streamlit Secrets or the "
            "environment (never hardcode it)."
        )
    return key


def get_google_api_key(explicit: str | None = None) -> str | None:
    """Resolve the Google Places key: explicit argument first, then the environment.

    Optional — when absent, Geoapify results are returned without Google-sourced
    phone/rating/review enrichment. Never hardcoded, never logged.
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    value = os.environ.get("GOOGLE_PLACES_API_KEY")
    return value.strip() if value and value.strip() else None


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------
def _get_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    """GET + JSON with uniform error handling (key is never surfaced in errors)."""
    try:
        resp = _SESSION.get(url, params=params, timeout=timeout)
    except requests.Timeout as exc:
        raise ApiRequestError(f"Geoapify request timed out after {timeout}s.") from exc
    except requests.RequestException as exc:
        raise ApiRequestError("Could not reach Geoapify (network error).") from exc

    if resp.status_code == 401:
        raise MissingApiKeyError("Geoapify rejected the API key (401 Unauthorized).")
    if resp.status_code != 200:
        raise ApiRequestError(f"Geoapify returned HTTP {resp.status_code}.")
    try:
        return resp.json()
    except ValueError as exc:
        raise ApiRequestError("Geoapify returned a non-JSON response.") from exc


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (fallback when the API omits distance)."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def directions_url(lat: float | None, lon: float | None) -> str:
    """Google Maps directions link to the destination coordinates."""
    if lat is None or lon is None:
        return ""
    return f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"


# -----------------------------------------------------------------------------
# Geocoding
# -----------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _geocode_cached(location: str, api_key: str, timeout: float) -> Coordinates:
    data = _get_json(
        GEOCODE_URL,
        {"text": location, "format": "json", "limit": 1, "apiKey": api_key},
        timeout,
    )
    results = data.get("results") or []
    if not results:
        raise LocationNotFoundError(f"Could not find a location matching '{location}'.")
    top = results[0]
    lat, lon = top.get("lat"), top.get("lon")
    if lat is None or lon is None:
        raise LocationNotFoundError(f"Could not resolve coordinates for '{location}'.")
    return Coordinates(lat=float(lat), lon=float(lon), formatted=top.get("formatted", ""))


def geocode(location: str, api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> Coordinates:
    """Convert a free-text location into coordinates via Geoapify Geocoding.

    Raises MissingApiKeyError, LocationNotFoundError, or ApiRequestError.
    """
    text = (location or "").strip()
    if not text:
        raise LocationNotFoundError("Enter a location to search for nearby resources.")
    key = _require_key(api_key)
    return _geocode_cached(text, key, timeout)


# -----------------------------------------------------------------------------
# Places
# -----------------------------------------------------------------------------
def _parse_feature(feature: dict[str, Any], resource_type: str, origin: Coordinates) -> Resource:
    props = feature.get("properties", {}) or {}
    lat = props.get("lat")
    lon = props.get("lon")
    # Geoapify returns `distance` (m) when a circle filter is used; else compute it.
    dist = props.get("distance")
    if dist is None and lat is not None and lon is not None:
        dist = haversine_m(origin.lat, origin.lon, float(lat), float(lon))

    address = props.get("address_line2") or props.get("formatted") or ""
    contact = props.get("contact", {}) or {}
    phone = contact.get("phone") or props.get("phone") or ""
    categories = props.get("categories") or []

    name = props.get("name") or (address.split(",")[0] if address else "") or "Unnamed facility"
    return Resource(
        name=name,
        type=resource_type,
        lat=None if lat is None else float(lat),
        lon=None if lon is None else float(lon),
        distance_m=None if dist is None else float(dist),
        address=address,
        phone=phone,
        category=categories[0] if categories else "",
        directions_url=directions_url(lat, lon),
    )


def find_resource_type(
    resource_type: str,
    origin: Coordinates,
    radius_m: int = DEFAULT_RADIUS_M,
    limit: int = DEFAULT_LIMIT,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[Resource]:
    """Nearby places of a single resource type, sorted by distance (nearest first)."""
    if resource_type not in RESOURCE_CATEGORIES:
        raise EmergencyServiceError(f"Unknown resource type '{resource_type}'.")
    key = _require_key(api_key)
    radius = max(MIN_RADIUS_M, min(int(radius_m), MAX_RADIUS_M))
    categories = ",".join(RESOURCE_CATEGORIES[resource_type])

    data = _get_json(
        PLACES_URL,
        {
            "categories": categories,
            "filter": f"circle:{origin.lon},{origin.lat},{radius}",
            "bias": f"proximity:{origin.lon},{origin.lat}",
            "limit": max(1, int(limit)),
            "apiKey": key,
        },
        timeout,
    )
    features = data.get("features") or []
    resources = [_parse_feature(f, resource_type, origin) for f in features]
    resources.sort(key=lambda r: (r.distance_m is None, r.distance_m or 0.0))
    return resources


# -----------------------------------------------------------------------------
# Google Places enrichment (optional) — phone, rating, reviews, listing URL
# -----------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def name_similarity(a: str, b: str) -> float:
    """Token Jaccard overlap between two place names (0..1)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _google_get(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    """GET a Google Places endpoint and map its body ``status`` to errors."""
    data = _get_json(url, params, timeout)
    status = data.get("status", "UNKNOWN")
    if status in ("OK", "ZERO_RESULTS"):
        return data
    msg = data.get("error_message", "")
    if status in ("OVER_QUERY_LIMIT", "REQUEST_DENIED"):
        raise GooglePlacesError(
            f"Google Places unavailable ({status}" + (f": {msg}" if msg else "")
            + "). Check the key, billing, and quota."
        )
    raise GooglePlacesError(f"Google Places error: {status}" + (f" ({msg})" if msg else ""))


def enrich_resource_google(resource: Resource, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> Resource:
    """Attach Google phone/rating/reviews/listing to ``resource`` in place.

    Matching is deliberately conservative: a candidate is accepted only when it is
    within ``GOOGLE_MATCH_RADIUS_M`` of the Geoapify point AND its name overlaps
    enough. On no confident match the Google fields are left empty ("Not
    available" at display time) — nothing is fabricated. Raises GooglePlacesError
    on key/quota/billing failures so the caller can report them once.
    """
    if resource.lat is None or resource.lon is None:
        return resource
    find = _google_get(
        GOOGLE_FIND_URL,
        {
            "input": resource.name,
            "inputtype": "textquery",
            "fields": "place_id,name,geometry",
            "locationbias": f"circle:{GOOGLE_MATCH_RADIUS_M}@{resource.lat},{resource.lon}",
            "key": api_key,
        },
        timeout,
    )
    candidates = find.get("candidates") or []
    if not candidates:
        return resource

    best = candidates[0]
    gname = best.get("name", "")
    gloc = (best.get("geometry") or {}).get("location") or {}
    glat, glon = gloc.get("lat"), gloc.get("lng")

    # Confident match gate: close enough AND similar name.
    if glat is not None and glon is not None:
        if haversine_m(resource.lat, resource.lon, float(glat), float(glon)) > GOOGLE_MATCH_RADIUS_M * 2:
            return resource
    if name_similarity(resource.name, gname) < NAME_MATCH_MIN:
        return resource

    place_id = best.get("place_id")
    if not place_id:
        return resource
    details = (_google_get(
        GOOGLE_DETAILS_URL,
        {"place_id": place_id, "fields": GOOGLE_DETAIL_FIELDS, "key": api_key},
        timeout,
    ).get("result")) or {}

    resource.google_matched = True
    resource.rating = details.get("rating")
    resource.reviews_count = details.get("user_ratings_total")
    resource.google_url = details.get("url", "")
    if not resource.phone:
        resource.phone = details.get("formatted_phone_number") or details.get("international_phone_number") or ""
    resource.reviews = [
        {
            "author": rv.get("author_name", ""),
            "rating": rv.get("rating"),
            "text": (rv.get("text") or "").strip(),
            "relative_time": rv.get("relative_time_description", ""),
        }
        for rv in (details.get("reviews") or [])[:MAX_REVIEWS]
    ]
    return resource


def find_nearby(
    location: str,
    resource_types: Iterable[str] = DEFAULT_RESOURCE_TYPES,
    radius_m: int = DEFAULT_RADIUS_M,
    limit: int = DEFAULT_LIMIT,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    google_api_key: str | None = None,
    google_enrich_limit: int = MAX_GOOGLE_ENRICH,
) -> SearchResult:
    """Geocode ``location`` then find nearby resources of each requested type.

    A failure for one resource type is recorded in ``result.errors`` and does not
    abort the others. Geocoding / key failures propagate (they affect everything).
    Empty result sets are returned as empty lists, not errors.

    When ``google_api_key`` is given, the nearest ``google_enrich_limit`` results
    per type are enriched with Google phone/rating/reviews. Any Google failure is
    recorded once under ``errors['_google']`` and never aborts the Geoapify data.
    """
    key = _require_key(api_key)
    origin = geocode(location, api_key=key, timeout=timeout)
    gkey = get_google_api_key(google_api_key)

    result = SearchResult(query_location=location.strip(), coordinates=origin, radius_m=int(radius_m))
    for rtype in resource_types:
        try:
            result.resources[rtype] = find_resource_type(
                rtype, origin, radius_m=radius_m, limit=limit, api_key=key, timeout=timeout
            )
        except EmergencyServiceError as exc:
            result.resources[rtype] = []
            result.errors[rtype] = str(exc)

    if gkey:
        for rtype, items in result.resources.items():
            for res in items[:max(0, int(google_enrich_limit))]:
                try:
                    enrich_resource_google(res, gkey, timeout=timeout)
                except GooglePlacesError as exc:
                    result.errors.setdefault("_google", str(exc))
                    break  # stop hitting a denied/over-quota endpoint
            if "_google" in result.errors:
                break
    return result
