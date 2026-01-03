def get_rate_limits():
    import httpx
    headers = {
        "Authorization": f"Bearer sk-proj-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"
    }
    resp = httpx.get("https://api.openai.com/v1/models", headers=headers)
    return {
        "limit": resp.headers.get("x-ratelimit-limit-requests"),
        "remaining": resp.headers.get("x-ratelimit-remaining-requests"),
        "reset": resp.headers.get("x-ratelimit-reset-requests"),
    }

print(get_rate_limits())
