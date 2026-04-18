from homie_core.mesh.inference_client import MeshInferenceClient


def test_client_generate_calls_handler():
    def fake_handler(prompt, max_tokens, temperature, stop):
        return f"response to: {prompt[:10]}"
    client = MeshInferenceClient(node_id="spoke-1", hub_node_id="hub-1", request_handler=fake_handler)
    assert client.generate("hello world", max_tokens=100) == "response to: hello worl"


def test_client_generate_with_stop():
    captured = {}
    def fake_handler(prompt, max_tokens, temperature, stop):
        captured["stop"] = stop
        return "ok"
    client = MeshInferenceClient(node_id="s1", hub_node_id="h1", request_handler=fake_handler)
    client.generate("test", stop=["END"])
    assert captured["stop"] == ["END"]


def test_client_is_available():
    assert MeshInferenceClient(node_id="s1", hub_node_id="h1", request_handler=lambda p, m, t, s: "ok").is_available is True
    assert MeshInferenceClient(node_id="s1", hub_node_id="h1", request_handler=None).is_available is False


def test_client_generate_raises_on_no_handler():
    client = MeshInferenceClient(node_id="s1", hub_node_id="h1", request_handler=None)
    try:
        client.generate("test")
        assert False
    except RuntimeError:
        pass


def test_client_stream_yields_tokens():
    client = MeshInferenceClient(node_id="s1", hub_node_id="h1", request_handler=lambda p, m, t, s: "streamed response")
    assert "".join(client.stream("test", max_tokens=50)) == "streamed response"
