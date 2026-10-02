"""Verify short-lived owner tokens minted by the Vercel login service."""
import base64
import hashlib
import hmac
import json
import math
import os
import time

def valid_token(token, secret=None, now=None):
    secret = os.environ.get('FLYTAIKO_AUTH_SECRET', '') if secret is None else secret
    if not secret or not token or len(token) > 2048:
        return False
    try:
        payload, signature = token.split('.')
        expected = base64.urlsafe_b64encode(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()).decode().rstrip('=')
        if not hmac.compare_digest(signature, expected):
            return False
        value = json.loads(base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4)))
        if not isinstance(value, dict):
            return False
        expiry = value.get('exp')
        return (value.get('sub') == 'owner' and value.get('aud') == 'gpu-worker'
                and isinstance(expiry, (int, float)) and math.isfinite(expiry) and expiry > (time.time() if now is None else now))
    except (ValueError, TypeError, UnicodeError):
        return False
