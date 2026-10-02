import base64
import hashlib
import hmac
import json
from flytaiko.owner_auth import valid_token

def token(value, key='test-secret'):
    body=base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip('=')
    signature=base64.urlsafe_b64encode(hmac.new(key.encode(),body.encode(),hashlib.sha256).digest()).decode().rstrip('=')
    return body+'.'+signature

def test_owner_only_expiring_audience_bound_tokens():
    good=token({'sub':'owner','aud':'gpu-worker','exp':200})
    assert valid_token(good,'test-secret',100)
    assert not valid_token(good,'test-secret',201)
    assert not valid_token(good+'x','test-secret',100)
    assert not valid_token(good,'wrong-key',100)
    assert not valid_token(good,'',100)
    assert not valid_token(token({'sub':'owner','aud':'web-owner','exp':200}),'test-secret',100)
    assert not valid_token(token({'sub':'visitor','aud':'gpu-worker','exp':200}),'test-secret',100)
