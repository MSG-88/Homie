"""Tests for LinkedIn AI content publisher."""
import pytest
from unittest.mock import MagicMock, patch
from homie_core.social_media.linkedin_publisher import (
    LinkedInPublisher, PostDraft, _AI_TOPICS,
)


class TestPostDraft:
    def test_to_linkedin_text(self):
        draft = PostDraft(
            topic="Local LLMs", title="Local LLMs", content="Great post content here.",
            hashtags=["AI", "LLM", "LocalAI"], category="thought_leadership",
        )
        text = draft.to_linkedin_text()
        assert "Great post content" in text
        assert "#AI" in text
        assert "#LLM" in text

    def test_character_limit(self):
        draft = PostDraft(
            topic="Test", title="Test", content="x" * 3500,
            hashtags=[], category="news",
        )
        assert len(draft.to_linkedin_text()) <= 3000


class TestLinkedInPublisher:
    def _mock_inference(self, **kwargs):
        prompt = kwargs.get("prompt", "")
        if "research" in prompt.lower():
            return "Key trends: 1. Local AI is growing. 2. Privacy matters. 3. Open source wins."
        return "AI is transforming how we work.\n\nHere's why local LLMs matter.\n\nHASHTAGS: AI, LLM, LocalAI"

    def test_get_topic_suggestions(self):
        pub = LinkedInPublisher(inference_fn=MagicMock(), profile_name="Test")
        topics = pub.get_topic_suggestions(5)
        assert len(topics) == 5
        for t in topics:
            assert t in _AI_TOPICS

    def test_avoids_repeat_topics(self):
        pub = LinkedInPublisher(inference_fn=MagicMock(), profile_name="Test")
        pub._post_history.append(_AI_TOPICS[0])
        topics = pub.get_topic_suggestions(5)
        assert _AI_TOPICS[0] not in topics

    def test_research_topic(self):
        pub = LinkedInPublisher(inference_fn=self._mock_inference, profile_name="Test")
        research = pub.research_topic("Local LLMs")
        assert "Local AI" in research or "trends" in research.lower()

    def test_generate_post(self):
        pub = LinkedInPublisher(inference_fn=self._mock_inference, profile_name="Test")
        draft = pub.generate_post("Local LLMs")
        assert isinstance(draft, PostDraft)
        assert draft.topic == "Local LLMs"
        assert draft.content
        assert len(draft.hashtags) > 0

    def test_generate_post_parses_hashtags(self):
        def mock_fn(**kwargs):
            return "Great post about AI.\n\nHASHTAGS: AI, MachineLearning, LLM"
        pub = LinkedInPublisher(inference_fn=mock_fn, profile_name="Test")
        draft = pub.generate_post("AI Trends")
        assert len(draft.hashtags) > 0

    def test_weekly_schedule(self):
        pub = LinkedInPublisher(inference_fn=MagicMock(), profile_name="Test")
        schedule = pub.generate_weekly_schedule(4)
        assert len(schedule) == 4
        for item in schedule:
            assert "week" in item
            assert "topic" in item
            assert "style" in item

    @patch("homie_core.social_media.linkedin_publisher.requests")
    def test_publish_success(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.headers = {"x-restli-id": "post123"}
        mock_resp.json.return_value = {"id": "post123"}
        mock_requests.post.return_value = mock_resp

        pub = LinkedInPublisher(inference_fn=MagicMock(), profile_name="Test")
        draft = PostDraft(topic="AI", title="AI", content="Content", hashtags=["AI"], category="news")
        result = pub.publish(draft, access_token="token123", person_id="abc")
        assert result["status"] == "published"

    @patch("homie_core.social_media.linkedin_publisher.requests")
    def test_publish_falls_back_to_ugc(self, mock_requests):
        # First call (rest API) fails, second (ugc) succeeds
        fail_resp = MagicMock()
        fail_resp.status_code = 403
        success_resp = MagicMock()
        success_resp.status_code = 201
        success_resp.json.return_value = {"id": "ugc123"}
        mock_requests.post.side_effect = [fail_resp, success_resp]

        pub = LinkedInPublisher(inference_fn=MagicMock(), profile_name="Test")
        draft = PostDraft(topic="AI", title="AI", content="Content", hashtags=[], category="news")
        result = pub.publish(draft, access_token="token123", person_id="abc")
        assert result["status"] == "published"
        assert result["api"] == "ugc"

    def test_generate_post_handles_inference_error(self):
        def fail_fn(**kwargs):
            raise RuntimeError("Model unavailable")
        pub = LinkedInPublisher(inference_fn=fail_fn, profile_name="Test")
        draft = pub.generate_post("AI Trends")
        assert "failed" in draft.content.lower()
