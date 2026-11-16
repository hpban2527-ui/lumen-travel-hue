from datetime import datetime, timedelta, timezone
import pytest
from firebase_client import AuthError, UserSession, register, login, clear_user_state, ensure_fresh_session


class FakeReply:
    status_code=200
    def json(self):
        return {'email':'test@example.com','localId':'uid123','idToken':'test_id',
                'refreshToken':'test_refresh','expiresIn':'3600'}


class Transport:
    def __init__(self): self.calls=[]
    def post(self, url, **kwargs):
        self.calls.append((url,kwargs))
        return FakeReply()


def test_firebase_sign_in_and_registration_call_correct_endpoints():
    t=Transport()
    assert register('public_api_key','test@example.com','password123',transport=t).uid=='uid123'
    assert login('public_api_key','test@example.com','password123',transport=t).email=='test@example.com'
    assert 'accounts:signUp' in t.calls[0][0]
    assert 'accounts:signInWithPassword' in t.calls[1][0]
    assert 'password123' in str(t.calls[0][1]['json'])


def test_validation_and_no_secret_configuration():
    with pytest.raises(AuthError,match='valid email'):
        login('key','not-an-email','password')
    with pytest.raises(AuthError,match='at least 6'):
        register('key','test@example.com','123')
    with pytest.raises(AuthError,match='not configured'):
        login('','test@example.com','password')


def test_session_reuse_and_expired_refresh():
    t=Transport()
    sess=UserSession('test@example.com','uid','id','refresh',datetime.now(timezone.utc)+timedelta(hours=1))
    assert ensure_fresh_session('key',sess,transport=t) is sess
    assert not t.calls


def test_signout_deletes_chat_and_tokens():
    state={'user':'token','messages':[1,2],'pending_question':'hello','custom':'private'}
    clear_user_state(state)
    assert not state


def test_invalid_firebase_response_raises_user_friendly_error():
    from firebase_client import UserSession
    for response in [{}, {"email": "a@example.com"}, {"email": "x", "localId": "x", "idToken": "x", "refreshToken": "x", "expiresIn": "?"}]:
        with pytest.raises(AuthError, match="invalid session"):
            UserSession.from_response(response)


def test_firebase_auth_emulator_endpoints_are_opt_in(monkeypatch):
    t = Transport()
    monkeypatch.setenv("FIREBASE_AUTH_EMULATOR_HOST", "127.0.0.1:9099")
    register("demo-local", "test@example.com", "password123", transport=t)
    login("demo-local", "test@example.com", "password123", transport=t)
    assert t.calls[0][0].startswith("http://127.0.0.1:9099/identitytoolkit.googleapis.com/v1/accounts:signUp")
    assert t.calls[1][0].startswith("http://127.0.0.1:9099/identitytoolkit.googleapis.com/v1/accounts:signInWithPassword")
    monkeypatch.delenv("FIREBASE_AUTH_EMULATOR_HOST")
    register("fake-public", "test@example.com", "password123", transport=t)
    assert t.calls[-1][0].startswith("https://identitytoolkit.googleapis.com")


def test_reject_nonlocal_auth_emulator(monkeypatch):
    monkeypatch.setenv("FIREBASE_AUTH_EMULATOR_HOST", "other-host:9099")
    with pytest.raises(AuthError, match="local Firebase Auth emulator"):
        register("demo-local", "test@example.com", "password123", transport=Transport())
