from unittest.mock import patch, MagicMock
from homie_core.screen_reader.ocr_reader import OCRReader
from homie_core.screen_reader.pii_filter import PIIFilter


class TestOCRReader:
    def test_init(self):
        reader = OCRReader(pii_filter=PIIFilter())
        assert reader is not None

    @patch("homie_core.screen_reader.ocr_reader.mss")
    def test_capture_returns_image(self, mock_mss):
        mock_sct = MagicMock()
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__ = MagicMock(return_value=False)
        mock_sct.grab.return_value = MagicMock(rgb=b"\x00" * (100 * 100 * 3), size=(100, 100))
        mock_mss.mss.return_value = mock_sct

        reader = OCRReader(pii_filter=PIIFilter())
        img = reader.capture_screen()
        assert img is not None

    def test_extract_text_filters_pii(self):
        reader = OCRReader(pii_filter=PIIFilter())
        # Mock OCR returning text with PII
        raw_text = "From: john@example.com\nSubject: Meeting"
        filtered = reader._apply_pii_filter(raw_text)
        assert "john@example.com" not in filtered
        assert "[EMAIL]" in filtered
