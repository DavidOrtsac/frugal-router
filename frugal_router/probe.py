"""Self-healing Fireworks bootstrap probe.

Seven scored submissions proved that guessing the judge environment's shape
wrong makes every remote call fail deterministically and silently: a wrong
model-ID namespace, an unexpected ALLOWED_MODELS serialization, a base-URL
path the SDK cannot join, or a polluted key all produce the same permanent
4xx on every attempt, and the router falls back to local answers with no
outward sign beyond a frozen score.

So the router stops assuming. At startup it probes tiny (max_tokens=1)
completions across the matrix of plausible shapes — base-URL variants x
model-ID forms x transport tweaks x auth shapes — and pins the first
combination that actually answers. Error classes steer the expansion: 401/403
adds alternate auth headers, SSL errors add verify=False, connect/proxy
errors add trust_env=False. If nothing answers within the budget, the router
degrades to local-only mode exactly as before: never worse, and every step is
logged to stderr for the one artifact we control.
"""

import os
import re
import sys
import time
from dataclasses import dataclass, field

_FIREWORKS_PREFIX = "accounts/fireworks/models/"
_PREFERENCE = ("kimi", "gemma", "minimax")
_CELL_TIMEOUT = 6.0


@dataclass(frozen=True)
class RemoteRuntime:
    """The pinned, probe-verified way to talk to the metering proxy."""

    ok: bool
    base_url: str = ""
    api_key: str = "EMPTY"
    headers: dict = field(default_factory=dict)  # extra auth headers, if any
    verify: bool = True
    trust_env: bool = True
    model_map: dict = field(default_factory=dict)  # short name -> working id
    note: str = ""


def _log(message: str) -> None:
    print(f"[frugal-router:probe] {message}", file=sys.stderr)


def base_url_variants(base: str) -> list:
    """Candidate bases in probe order: as-published, then +/v1, then
    +/inference/v1 (skipping variants the base already ends with)."""
    base = (base or "").strip().strip("'\"").rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")].rstrip("/")
    if base and "://" not in base:
        base = "https://" + base
    if not base:
        return []
    variants = [base]
    if not base.endswith("/v1"):
        variants.append(base + "/v1")
    if not base.endswith("/inference/v1"):
        variants.append(base + "/inference/v1")
    return list(dict.fromkeys(variants))


def model_id_forms(token: str) -> list:
    """The three ways a model can be addressed: exactly as published, as the
    bare short name, and as the canonical Fireworks serverless path."""
    short = token.rsplit("/", 1)[-1]
    return list(dict.fromkeys([token, short, _FIREWORKS_PREFIX + short]))


def key_candidates(config_key: str) -> list:
    """The sanitized configured key, any env var that looks like a Fireworks
    key under another name, and 'EMPTY' as the no-auth last resort."""
    keys = []
    cleaned = (config_key or "").strip().strip("'\"")
    if cleaned:
        keys.append(cleaned)
    for name, value in os.environ.items():
        if re.search(r"FIREWORKS.*(KEY|TOKEN)", name, re.IGNORECASE):
            candidate = (value or "").strip().strip("'\"")
            if candidate and candidate not in keys:
                keys.append(candidate)
    if not keys:
        keys.append("EMPTY")
    return keys


def ordered_model_candidates(allowed: tuple, raw: tuple) -> list:
    """Preference order kimi > gemma > minimax > published order, with the
    unsanitized raw tokens appended as last-resort candidates."""
    ordered = []
    for pattern in _PREFERENCE:
        for token in allowed:
            if pattern in token.lower() and token not in ordered:
                ordered.append(token)
    for token in allowed:
        if token not in ordered:
            ordered.append(token)
    for token in raw:
        if token not in ordered:
            ordered.append(token)
    return ordered


def _probe_once(client, base: str, model: str, key: str, headers: dict):
    """One probe cell. Returns (ok, http_status_or_None, exception_or_None)."""
    url = base.rstrip("/") + "/chat/completions"
    request_headers = {"Authorization": f"Bearer {key}"}
    request_headers.update(headers or {})
    try:
        response = client.post(url, json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 1,
            "temperature": 0,
        }, headers=request_headers)
    except Exception as exc:
        return False, None, exc
    if response.status_code == 200:
        try:
            if response.json().get("choices"):
                return True, 200, None
        except ValueError:
            pass
        return False, 200, None
    return False, response.status_code, None


def bootstrap_remote(config, budget_seconds: float = 75.0) -> RemoteRuntime:
    """Find a working (base URL, key, transport, auth, model id) combination.

    Phase 1 sweeps for ANY working channel using the top-preference model's
    forms. Phase 2 resolves every distinct model the routing table can emit
    through the pinned channel, demoting unresolvable models to the next
    preference that answers."""
    import httpx

    started = time.monotonic()
    bases = base_url_variants(config.fireworks_base_url)
    keys = key_candidates(config.fireworks_api_key)
    raw_tokens = getattr(config, "allowed_models_raw", ()) or ()
    candidates = ordered_model_candidates(config.allowed_models, raw_tokens)

    _log(f"base={config.fireworks_base_url!r} variants={bases}")
    _log(f"key_present={bool(config.fireworks_api_key)} "
         f"key_len={len(config.fireworks_api_key or '')} key_candidates={len(keys)}")
    _log(f"allowed={list(config.allowed_models)} raw_extra={list(raw_tokens)}")

    if not bases or not candidates:
        _log("probe=ALL_FAILED (no base URL or no model candidates)")
        return RemoteRuntime(ok=False, note="no base or candidates")

    channel = None
    channel_model = None
    # Sweep with EVERY distinct allowed model, not just the favorite: a proxy
    # where only one obscure model answers must never read as "no channel".
    # Budget checks bound the worst case; a working channel answers early.
    probe_forms = []
    seen_shorts = set()
    for token in candidates:
        short = token.rsplit("/", 1)[-1]
        if short in seen_shorts:
            continue
        seen_shorts.add(short)
        probe_forms.extend(model_id_forms(token))
    probe_forms = list(dict.fromkeys(probe_forms))
    transports = [{"verify": True, "trust_env": True}]
    auth_shapes = [{}]
    expanded = {"auth": False, "ssl": False, "proxy": False}

    queue = [(b, 0, k) for b in bases for k in keys]
    index = 0
    while index < len(queue) and channel is None:
        if time.monotonic() - started > budget_seconds:
            _log("probe budget exhausted during channel sweep")
            break
        base, transport_index, key = queue[index]
        index += 1
        transport = transports[transport_index]
        cell_hopeless = False
        cell_timeouts = 0
        with httpx.Client(timeout=_CELL_TIMEOUT, **transport) as client:
            for auth in list(auth_shapes):
                if cell_hopeless:
                    break
                for model in probe_forms:
                    if time.monotonic() - started > budget_seconds:
                        break
                    ok, status, exc = _probe_once(client, base, model, key, auth)
                    outcome = ("OK" if ok else
                               str(status) if status is not None else
                               exc.__class__.__name__)
                    _log(f"cell base={base} model={model} "
                         f"auth={'alt' if auth else 'bearer'} "
                         f"verify={transport['verify']} "
                         f"trust_env={transport['trust_env']} -> {outcome}")
                    if ok:
                        channel = {"base": base, "key": key,
                                   "transport": transport, "auth": auth}
                        channel_model = model
                        break
                    if status in (401, 403) and not expanded["auth"]:
                        expanded["auth"] = True
                        auth_shapes.append({"api-key": key, "x-api-key": key})
                        # Re-enqueue the CURRENT cell so the alternate auth is
                        # tried against this base too, not just later ones.
                        queue.append((base, transport_index, key))
                    if exc is not None:
                        name = exc.__class__.__name__.lower()
                        detail = str(exc).lower()
                        if (("ssl" in name or "certificate" in name or
                             "ssl" in detail) and not expanded["ssl"]):
                            expanded["ssl"] = True
                            transports.append(
                                {"verify": False,
                                 "trust_env": transport["trust_env"]})
                            queue.extend((b, len(transports) - 1, k)
                                         for b in bases for k in keys)
                        if (("connect" in name or "proxy" in name)
                                and not expanded["proxy"]):
                            expanded["proxy"] = True
                            transports.append(
                                {"verify": transport["verify"],
                                 "trust_env": False})
                            queue.extend((b, len(transports) - 1, k)
                                         for b in bases for k in keys)
                        # Host-level failures are model-independent: trying
                        # more model IDs against a host that will not connect
                        # only starves the expansions of probe budget.
                        if ("connect" in name or "proxy" in name
                                or "ssl" in name or "certificate" in name):
                            cell_hopeless = True
                            break
                        if "timeout" in name:
                            cell_timeouts += 1
                            if cell_timeouts >= 2:
                                cell_hopeless = True
                                break
                if channel:
                    break

    if channel is None:
        _log("probe=ALL_FAILED — router will run local-only")
        return RemoteRuntime(ok=False, note="ALL_FAILED")

    _log(f"channel PINNED base={channel['base']} model={channel_model} "
         f"auth={'alt' if channel['auth'] else 'bearer'} "
         f"verify={channel['transport']['verify']} "
         f"trust_env={channel['transport']['trust_env']}")

    model_map = {channel_model.rsplit("/", 1)[-1]: channel_model}
    distinct_preferred = list(dict.fromkeys(config.remote_by_category.values()))
    with httpx.Client(timeout=_CELL_TIMEOUT, **channel["transport"]) as client:
        for preferred in distinct_preferred:
            short = preferred.rsplit("/", 1)[-1]
            if short in model_map:
                continue
            if time.monotonic() - started > budget_seconds:
                _log(f"budget exhausted; {short} unresolved (falls back to "
                     f"channel model)")
                continue
            matching = [t for t in candidates
                        if t.rsplit("/", 1)[-1] == short]
            fallbacks = [t for t in candidates if t not in matching]
            resolved = None
            for token in matching + fallbacks:
                for form in model_id_forms(token):
                    if time.monotonic() - started > budget_seconds:
                        break
                    ok, status, exc = _probe_once(
                        client, channel["base"], form, channel["key"],
                        channel["auth"])
                    if ok:
                        resolved = form
                        break
                if resolved or time.monotonic() - started > budget_seconds:
                    break
            if resolved:
                model_map[short] = resolved
                _log(f"model {preferred} -> {resolved}")
            else:
                model_map[short] = channel_model
                _log(f"model {preferred} UNRESOLVED -> using channel model "
                     f"{channel_model}")

    return RemoteRuntime(
        ok=True,
        base_url=channel["base"],
        api_key=channel["key"],
        headers=dict(channel["auth"]),
        verify=channel["transport"]["verify"],
        trust_env=channel["transport"]["trust_env"],
        model_map=model_map,
        note=f"pinned in {time.monotonic() - started:.1f}s",
    )
