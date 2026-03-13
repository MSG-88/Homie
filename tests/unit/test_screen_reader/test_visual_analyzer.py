from unittest.mock import patch, MagicMock, AsyncMock
from homie_core.screen_reader.visual_analyzer import VisualAnalyzer


class TestVisualAnalyzer:
    def test_init_cloud_mode(self):
        analyzer = VisualAnalyzer(engine="cloud", api_base_url="https://api.qubrid.com")
        assert analyzer._engine == "cloud"

    def test_init_local_mode(self):
        analyzer = VisualAnalyzer(engine="local")
        assert analyzer._engine == "local"

    def test_resize_image(self):
        from PIL import Image
        import io
        analyzer = VisualAnalyzer(engine="local")
        img = Image.new("RGB", (1920, 1080))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resized = analyzer._resize(buf.getvalue(), max_height=720)
        result_img = Image.open(io.BytesIO(resized))
        assert result_img.height <= 720

    @patch("homie_core.screen_reader.visual_analyzer.requests.post")
    def test_analyze_cloud(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "User is editing Python in VS Code"}}]},
        )
        analyzer = VisualAnalyzer(engine="cloud", api_base_url="https://api.qubrid.com")
        result = analyzer.analyze(b"fake_image_bytes")
        assert "VS Code" in result
        assert isinstance(result, str)
