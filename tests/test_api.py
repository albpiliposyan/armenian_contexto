from unittest.mock import Mock

from fastapi.testclient import TestClient

from armenian_contexto.api import create_app


def mock_engine():
    engine = Mock()
    engine.words = ["դպրոց", "ուսուցիչ", "մեքենա"]
    engine.get_rank.side_effect = lambda guess, target: {
        "guess": guess,
        "processed_guess": guess,
        "target": target,
        "similarity": 0.5,
        "score": 0.5,
        "rank": 3,
        "is_correct": False,
        "components": {
            "armenian_fasttext": 0.4,
            "same_category": 1.0,
            "same_pos": 1.0,
            "english_similarity": 0.8,
            "final_score": 0.5,
        },
    }
    engine.closest_words.return_value = [
        {"word": "դպրոց", "similarity": 1.0, "score": 1.0, "rank": 1}
    ]
    return engine


def test_health_loads_engine_once_at_startup():
    engine = mock_engine()
    engine_factory = Mock(return_value=engine)
    app = create_app(engine_factory=engine_factory)

    with TestClient(app) as client:
        first_response = client.get("/api/health")
        second_response = client.get("/api/health")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["engine_loaded_once"] is True
    assert first_response.json()["vocabulary_size"] == 3
    engine_factory.assert_called_once()


def test_guess_endpoint_uses_engine():
    engine = mock_engine()
    app = create_app(engine=engine)

    with TestClient(app) as client:
        response = client.get(
            "/api/guess",
            params={"target": "դպրոց", "guess": "ուսուցիչ"},
        )

    assert response.status_code == 200
    assert response.json()["rank"] == 3
    engine.get_rank.assert_called_once_with("ուսուցիչ", "դպրոց")


def test_closest_endpoint_uses_engine():
    engine = mock_engine()
    app = create_app(engine=engine)

    with TestClient(app) as client:
        response = client.get(
            "/api/closest",
            params={"target": "դպրոց", "top_k": 1},
        )

    assert response.status_code == 200
    assert response.json()["closest"][0]["word"] == "դպրոց"
    engine.closest_words.assert_called_once_with("դպրոց", 1)


def test_batch_guess_endpoint_runs_multiple_engine_calls():
    engine = mock_engine()
    app = create_app(engine=engine)

    with TestClient(app) as client:
        response = client.post(
            "/api/batch-guess",
            json={"target": "դպրոց", "guesses": ["ուսուցիչ", "մեքենա"]},
        )

    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert [item["guess"] for item in response.json()["results"]] == [
        "ուսուցիչ",
        "մեքենա",
    ]
    assert engine.get_rank.call_count == 2


def test_missing_model_error_is_returned_as_service_unavailable():
    engine = mock_engine()
    engine.get_rank.side_effect = FileNotFoundError("missing model")
    app = create_app(engine=engine)

    with TestClient(app) as client:
        response = client.get(
            "/api/guess",
            params={"target": "դպրոց", "guess": "անհայտ"},
        )

    assert response.status_code == 503
    assert "missing model" in response.json()["detail"]
