"""Tests for the Geoapify emergency-resources backend (no live API calls).

All HTTP is mocked, so these run offline and require no GEOAPIFY_API_KEY.

Run with:  python -m pytest -q
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import emergency_services as emsvc  # noqa: E402


# --- fake HTTP layer ---------------------------------------------------------
class FakeResp:
    def __init__(self, status=200, payload=None, raise_exc=None):
        self.status_code = status
        self._payload = payload or {}
        self._raise = raise_exc

    def json(self):
        return self._payload


def _patch_get(monkeypatch, handler):
    """Replace the module session's .get with a handler(url, params, timeout)."""
    monkeypatch.setattr(emsvc._SESSION, "get", lambda url, params, timeout: handler(url, params))
    emsvc._geocode_cached.cache_clear()  # avoid cross-test cache bleed


GEO_PAYLOAD = {"results": [{"lat": 28.6329, "lon": 77.2195, "formatted": "Connaught Place, New Delhi"}]}


def _places_payload(*features):
    return {"features": list(features)}


def _feature(name, lat, lon, dist=None, phone=None, cat="service.police"):
    props = {"name": name, "lat": lat, "lon": lon, "categories": [cat],
             "formatted": f"{name}, somewhere", "address_line2": f"{name}, somewhere"}
    if dist is not None:
        props["distance"] = dist
    if phone is not None:
        props["contact"] = {"phone": phone}
    return {"properties": props}


# --- key handling ------------------------------------------------------------
def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEOAPIFY_API_KEY", raising=False)
    with pytest.raises(emsvc.MissingApiKeyError):
        emsvc.geocode("Delhi")


def test_explicit_key_preferred(monkeypatch):
    monkeypatch.delenv("GEOAPIFY_API_KEY", raising=False)
    assert emsvc.get_api_key("  abc  ") == "abc"


def test_env_key_fallback(monkeypatch):
    monkeypatch.setenv("GEOAPIFY_API_KEY", "envkey")
    assert emsvc.get_api_key() == "envkey"


# --- geocoding ---------------------------------------------------------------
def test_geocode_success(monkeypatch):
    _patch_get(monkeypatch, lambda url, params: FakeResp(payload=GEO_PAYLOAD))
    c = emsvc.geocode("Connaught Place", api_key="k")
    assert c.lat == pytest.approx(28.6329) and c.lon == pytest.approx(77.2195)
    assert "Connaught" in c.formatted


def test_geocode_empty_is_not_found(monkeypatch):
    _patch_get(monkeypatch, lambda url, params: FakeResp(payload={"results": []}))
    with pytest.raises(emsvc.LocationNotFoundError):
        emsvc.geocode("asldkfjzzz", api_key="k")


def test_blank_location_rejected():
    with pytest.raises(emsvc.LocationNotFoundError):
        emsvc.geocode("   ", api_key="k")


# --- HTTP error mapping ------------------------------------------------------
def test_timeout_mapped(monkeypatch):
    def boom(url, params, timeout):
        raise emsvc.requests.Timeout()
    monkeypatch.setattr(emsvc._SESSION, "get", boom)
    with pytest.raises(emsvc.ApiRequestError):
        emsvc._get_json("u", {}, 1)


def test_401_mapped_to_missing_key(monkeypatch):
    monkeypatch.setattr(emsvc._SESSION, "get", lambda url, params, timeout: FakeResp(status=401))
    with pytest.raises(emsvc.MissingApiKeyError):
        emsvc._get_json("u", {}, 1)


def test_non_200_mapped(monkeypatch):
    monkeypatch.setattr(emsvc._SESSION, "get", lambda url, params, timeout: FakeResp(status=500))
    with pytest.raises(emsvc.ApiRequestError):
        emsvc._get_json("u", {}, 1)


# --- places parsing & sorting ------------------------------------------------
def test_find_resource_type_sorts_by_distance(monkeypatch):
    far = _feature("Far Station", 28.70, 77.30, dist=3000)
    near = _feature("Near Station", 28.64, 77.22, dist=400)
    monkeypatch.setattr(emsvc._SESSION, "get",
                        lambda url, params, timeout: FakeResp(payload=_places_payload(far, near)))
    origin = emsvc.Coordinates(28.6329, 77.2195)
    out = emsvc.find_resource_type("police", origin, api_key="k")
    assert [r.name for r in out] == ["Near Station", "Far Station"]
    assert out[0].distance_km == 0.4
    assert out[0].directions_url.endswith("destination=28.64,77.22")


def test_distance_fallback_when_api_omits_it(monkeypatch):
    f = _feature("No-Dist Hospital", 28.6329, 77.2195, dist=None, cat="healthcare.hospital")
    monkeypatch.setattr(emsvc._SESSION, "get",
                        lambda url, params, timeout: FakeResp(payload=_places_payload(f)))
    origin = emsvc.Coordinates(28.6329, 77.2195)
    out = emsvc.find_resource_type("hospital", origin, api_key="k")
    assert out[0].distance_m is not None and out[0].distance_m < 50  # ~same point


def test_phone_and_address_parsed(monkeypatch):
    f = _feature("City Clinic", 28.64, 77.22, dist=100, phone="+91-11-12345678",
                 cat="healthcare.clinic_or_praxis")
    monkeypatch.setattr(emsvc._SESSION, "get",
                        lambda url, params, timeout: FakeResp(payload=_places_payload(f)))
    out = emsvc.find_resource_type("clinic", emsvc.Coordinates(28.64, 77.22), api_key="k")
    assert out[0].phone == "+91-11-12345678"
    assert "City Clinic" in out[0].address


def test_unknown_resource_type_raises():
    with pytest.raises(emsvc.EmergencyServiceError):
        emsvc.find_resource_type("dragon_lair", emsvc.Coordinates(0, 0), api_key="k")


def test_radius_is_clamped(monkeypatch):
    captured = {}

    def handler(url, params, timeout):
        captured["filter"] = params["filter"]
        return FakeResp(payload=_places_payload())
    monkeypatch.setattr(emsvc._SESSION, "get", handler)
    emsvc.find_resource_type("police", emsvc.Coordinates(1, 2), radius_m=10_000_000, api_key="k")
    assert f"{emsvc.MAX_RADIUS_M}" in captured["filter"]  # clamped to max


# --- find_nearby orchestration ----------------------------------------------
def test_find_nearby_collects_types_and_isolates_errors(monkeypatch):
    emsvc._geocode_cached.cache_clear()

    def handler(url, params):
        if url == emsvc.GEOCODE_URL:
            return FakeResp(payload=GEO_PAYLOAD)
        # police ok, hospital errors (simulated 500)
        if params.get("categories", "").startswith("service.police"):
            return FakeResp(payload=_places_payload(_feature("PS One", 28.64, 77.22, dist=200)))
        return FakeResp(status=500)
    monkeypatch.setattr(emsvc._SESSION, "get", lambda url, params, timeout: handler(url, params))

    result = emsvc.find_nearby("Delhi", resource_types=("police", "hospital"), api_key="k")
    assert result.total == 1
    assert result.resources["police"][0].name == "PS One"
    assert result.resources["hospital"] == []
    assert "hospital" in result.errors  # one type failed, others survived
    assert result.is_empty is False


def test_find_nearby_empty_results_not_an_error(monkeypatch):
    emsvc._geocode_cached.cache_clear()

    def handler(url, params):
        if url == emsvc.GEOCODE_URL:
            return FakeResp(payload=GEO_PAYLOAD)
        return FakeResp(payload=_places_payload())
    monkeypatch.setattr(emsvc._SESSION, "get", lambda url, params, timeout: handler(url, params))
    result = emsvc.find_nearby("Delhi", resource_types=("police",), api_key="k")
    assert result.is_empty is True
    assert result.errors == {}


# --- Google Places enrichment ------------------------------------------------
def _google_find(name, lat, lon, place_id="pid1"):
    return {"status": "OK", "candidates": [
        {"place_id": place_id, "name": name, "geometry": {"location": {"lat": lat, "lng": lon}}}]}


def _google_details(**fields):
    return {"status": "OK", "result": fields}


def test_name_similarity_basic():
    assert emsvc.name_similarity("Delhi Police Station", "Delhi Police Stn") > 0.3
    assert emsvc.name_similarity("Apollo Hospital", "City Pharmacy") == 0.0


def test_get_google_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    assert emsvc.get_google_api_key("  k  ") == "k"
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "envg")
    assert emsvc.get_google_api_key() == "envg"


def test_enrich_success(monkeypatch):
    def handler(url, params, timeout):
        if url == emsvc.GOOGLE_FIND_URL:
            return FakeResp(payload=_google_find("Apollo Hospital", 28.6329, 77.2195))
        return FakeResp(payload=_google_details(
            rating=4.3, user_ratings_total=120, url="https://maps.google.com/?cid=1",
            formatted_phone_number="011 2222 3333",
            reviews=[{"author_name": "A", "rating": 5, "text": "Good", "relative_time_description": "a week ago"}]))
    monkeypatch.setattr(emsvc._SESSION, "get", handler)
    res = emsvc.Resource(name="Apollo Hospital", type="hospital", lat=28.6329, lon=77.2195)
    emsvc.enrich_resource_google(res, "gkey")
    assert res.google_matched is True
    assert res.rating == 4.3 and res.reviews_count == 120
    assert res.phone == "011 2222 3333"
    assert res.google_url.startswith("https://maps.google")
    assert len(res.reviews) == 1 and res.reviews[0]["author"] == "A"


def test_enrich_rejects_distant_candidate(monkeypatch):
    # Candidate ~5km away → beyond 2x match radius → no match, nothing fabricated.
    monkeypatch.setattr(emsvc._SESSION, "get",
                        lambda url, params, timeout: FakeResp(payload=_google_find("Apollo Hospital", 28.70, 77.30)))
    res = emsvc.Resource(name="Apollo Hospital", type="hospital", lat=28.6329, lon=77.2195)
    emsvc.enrich_resource_google(res, "gkey")
    assert res.google_matched is False and res.rating is None


def test_enrich_rejects_name_mismatch(monkeypatch):
    monkeypatch.setattr(emsvc._SESSION, "get",
                        lambda url, params, timeout: FakeResp(payload=_google_find("Totally Different Cafe", 28.6329, 77.2195)))
    res = emsvc.Resource(name="Apollo Hospital", type="hospital", lat=28.6329, lon=77.2195)
    emsvc.enrich_resource_google(res, "gkey")
    assert res.google_matched is False


def test_enrich_no_candidates_leaves_empty(monkeypatch):
    monkeypatch.setattr(emsvc._SESSION, "get",
                        lambda url, params, timeout: FakeResp(payload={"status": "ZERO_RESULTS", "candidates": []}))
    res = emsvc.Resource(name="Apollo Hospital", type="hospital", lat=28.6329, lon=77.2195)
    emsvc.enrich_resource_google(res, "gkey")
    assert res.google_matched is False and res.rating is None


def test_google_quota_error_raised(monkeypatch):
    monkeypatch.setattr(emsvc._SESSION, "get",
                        lambda url, params, timeout: FakeResp(payload={"status": "OVER_QUERY_LIMIT", "error_message": "quota"}))
    res = emsvc.Resource(name="Apollo Hospital", type="hospital", lat=28.6329, lon=77.2195)
    with pytest.raises(emsvc.GooglePlacesError):
        emsvc.enrich_resource_google(res, "gkey")


def test_find_nearby_records_google_error_but_keeps_results(monkeypatch):
    emsvc._geocode_cached.cache_clear()

    def handler(url, params):
        if url == emsvc.GEOCODE_URL:
            return FakeResp(payload=GEO_PAYLOAD)
        if url == emsvc.PLACES_URL:
            return FakeResp(payload=_places_payload(_feature("PS One", 28.6329, 77.2195, dist=100)))
        # any Google call is denied
        return FakeResp(payload={"status": "REQUEST_DENIED", "error_message": "billing"})
    monkeypatch.setattr(emsvc._SESSION, "get", lambda url, params, timeout: handler(url, params))

    result = emsvc.find_nearby("Delhi", resource_types=("police",), api_key="k", google_api_key="g")
    assert result.resources["police"][0].name == "PS One"    # Geoapify data survives
    assert "_google" in result.errors                         # google failure recorded once


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
