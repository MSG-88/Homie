from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from homie_core.neural.model_manager import EmbeddingModel


def test_model_not_loaded_initially():
    model = EmbeddingModel()
    assert not model.is_loaded


def test_embed_raises_when_not_loaded():
    model = EmbeddingModel()
    try:
        model.embed("hello")
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()


def test_embed_batch_raises_when_not_loaded():
    model = EmbeddingModel()
    try:
        model.embed_batch(["hello"])
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()


def test_load_and_embed(tmp_path):
    """Test with a mock ONNX session."""
    model = EmbeddingModel(cache_dir=tmp_path)
    fake_output = np.random.randn(1, 384).astype(np.float32)

    with patch.object(model, "_load_onnx_session") as mock_load:
        mock_session = MagicMock()
        mock_session.run.return_value = [fake_output]
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        mock_load.return_value = mock_session

        with patch.object(model, "_load_tokenizer") as mock_tok:
            mock_tokenizer = MagicMock()
            encoding = MagicMock()
            encoding.ids = list(range(10))
            encoding.attention_mask = [1] * 10
            mock_tokenizer.encode.return_value = encoding
            mock_tok.return_value = mock_tokenizer

            model.load()

    assert model.is_loaded
    assert model.dimension == 384

    result = model.embed("test sentence")
    assert len(result) == 384
    assert isinstance(result[0], float)


def test_embed_batch(tmp_path):
    model = EmbeddingModel(cache_dir=tmp_path)
    fake_output = np.random.randn(2, 384).astype(np.float32)

    with patch.object(model, "_load_onnx_session") as mock_load:
        mock_session = MagicMock()
        mock_session.run.return_value = [fake_output]
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        mock_load.return_value = mock_session

        with patch.object(model, "_load_tokenizer") as mock_tok:
            mock_tokenizer = MagicMock()
            encoding = MagicMock()
            encoding.ids = list(range(10))
            encoding.attention_mask = [1] * 10
            mock_tokenizer.encode.return_value = encoding
            mock_tok.return_value = mock_tokenizer

            model.load()

    results = model.embed_batch(["hello", "world"])
    assert len(results) == 2
    assert len(results[0]) == 384


def test_unload():
    model = EmbeddingModel()
    model._session = MagicMock()
    model._tokenizer = MagicMock()
    model._loaded = True
    model._dimension = 384

    model.unload()
    assert not model.is_loaded
    assert model._session is None
