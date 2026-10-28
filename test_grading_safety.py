import study


def test_record_attempt_does_not_use_real_database(monkeypatch):

    def fake_get_conn():
        raise RuntimeError("Fake database was called")

    monkeypatch.setattr(study, "get_conn", fake_get_conn)