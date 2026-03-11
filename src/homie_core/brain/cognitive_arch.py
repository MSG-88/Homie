"""Cognitive Architecture — Homie's intelligent response pipeline.

Replaces the dumb "compress prompt → generate" pipe with a multi-stage
reasoning system that uses ALL of Homie's intelligence infrastructure:

Stage 1: PERCEIVE  — Gather full situational awareness from all sensors
Stage 2: CLASSIFY  — Determine query complexity and required depth
Stage 3: RETRIEVE  — Pull relevant memories using query-aware scoring
Stage 4: REASON    — Build a rich, structured prompt with cognitive context
Stage 5: REFLECT   — Score response confidence via Platt scaling
Stage 6: ADAPT     — Adjust response strategy based on user state

Algorithms:
- TF-IDF relevance scoring for memory retrieval
- Query complexity classification via lexical + structural heuristics
- Adaptive token budgeting based on complexity class
- Cognitive load estimation from flow state + task duration
- Response strategy selection via self-reflection confidence
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from homie_core.memory.working import WorkingMemory
from homie_core.memory.episodic import EpisodicMemory
from homie_core.memory.semantic import SemanticMemory
from homie_core.intelligence.self_reflection import SelfReflection


# ---------------------------------------------------------------------------
# Query complexity classification
# ---------------------------------------------------------------------------

class QueryComplexity:
    TRIVIAL = "trivial"      # "hi", "thanks", single-word
    SIMPLE = "simple"        # factual lookup, short answer
    MODERATE = "moderate"    # requires context + some reasoning
    COMPLEX = "complex"      # multi-step reasoning, synthesis
    DEEP = "deep"            # needs planning, full context, long response


# Heuristic signals for complexity classification
_COMPLEX_MARKERS = re.compile(
    r"\b(how|why|explain|compare|analyze|design|plan|implement|debug|"
    r"what if|should i|trade.?off|difference between|pros and cons|"
    r"step by step|walk me through|help me understand)\b",
    re.IGNORECASE,
)
_TRIVIAL_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|bye|goodbye|good morning|"
    r"good night|sup|yo|hmm|hm|lol|haha|cool|nice|great|sure|yep|nope)[\s!.?]*$",
    re.IGNORECASE,
)


def classify_query_complexity(text: str, conversation_depth: int = 0) -> str:
    """Classify query complexity using lexical + structural heuristics.

    Returns one of: trivial, simple, moderate, complex, deep.
    """
    text = text.strip()
    words = text.split()
    n_words = len(words)

    # Trivial: greetings, single-word, very short
    if _TRIVIAL_PATTERNS.match(text):
        return QueryComplexity.TRIVIAL
    if n_words <= 2 and not _COMPLEX_MARKERS.search(text):
        return QueryComplexity.TRIVIAL

    # Count complexity signals
    complex_matches = len(_COMPLEX_MARKERS.findall(text))
    has_question_mark = "?" in text
    has_code_markers = any(c in text for c in ["`", "```", "def ", "class ", "import "])
    n_sentences = max(1, len(re.split(r"[.!?]+", text)))
    has_multiple_clauses = n_sentences >= 3 or text.count(",") >= 3

    # Score it
    score = 0.0
    score += min(n_words / 30.0, 1.0) * 2.0          # longer = more complex
    score += complex_matches * 1.5                      # explicit complexity markers
    score += 1.0 if has_question_mark else 0.0
    score += 1.5 if has_code_markers else 0.0
    score += 1.0 if has_multiple_clauses else 0.0
    score += min(conversation_depth / 5.0, 1.0) * 0.5  # deeper conversations need more context

    if score < 1.5:
        return QueryComplexity.SIMPLE
    elif score < 3.5:
        return QueryComplexity.MODERATE
    elif score < 6.0:
        return QueryComplexity.COMPLEX
    else:
        return QueryComplexity.DEEP


# ---------------------------------------------------------------------------
# Adaptive token budgets per complexity class
# ---------------------------------------------------------------------------

_TOKEN_BUDGETS = {
    QueryComplexity.TRIVIAL:  {"max_tokens": 128,  "prompt_chars": 1000,  "temperature": 0.8},
    QueryComplexity.SIMPLE:   {"max_tokens": 256,  "prompt_chars": 2000,  "temperature": 0.7},
    QueryComplexity.MODERATE: {"max_tokens": 512,  "prompt_chars": 4000,  "temperature": 0.7},
    QueryComplexity.COMPLEX:  {"max_tokens": 1024, "prompt_chars": 6000,  "temperature": 0.6},
    QueryComplexity.DEEP:     {"max_tokens": 2048, "prompt_chars": 8000,  "temperature": 0.5},
}


# ---------------------------------------------------------------------------
# TF-IDF relevance scoring for memory retrieval
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r"\w+", text.lower())


def _tf_idf_relevance(query: str, documents: list[str]) -> list[float]:
    """Score documents against query using TF-IDF cosine similarity.

    Lightweight pure-Python implementation — no external deps.
    """
    query_tokens = _tokenize(query)
    if not query_tokens or not documents:
        return [0.0] * len(documents)

    # Build document frequency
    doc_tokens = [_tokenize(d) for d in documents]
    n_docs = len(documents)
    df: Counter = Counter()
    for tokens in doc_tokens:
        for word in set(tokens):
            df[word] += 1

    # IDF: log(N / df) with smoothing
    idf = {word: math.log((n_docs + 1) / (count + 1)) + 1.0 for word, count in df.items()}

    # Query TF-IDF vector
    q_tf = Counter(query_tokens)
    q_vec = {w: q_tf[w] * idf.get(w, 1.0) for w in q_tf}
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    # Score each document
    scores = []
    for tokens in doc_tokens:
        d_tf = Counter(tokens)
        d_vec = {w: d_tf[w] * idf.get(w, 1.0) for w in d_tf}
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0

        # Cosine similarity
        dot = sum(q_vec.get(w, 0.0) * d_vec.get(w, 0.0) for w in set(q_vec) | set(d_vec))
        scores.append(dot / (q_norm * d_norm))

    return scores


# ---------------------------------------------------------------------------
# Situational Awareness (Stage 1: PERCEIVE)
# ---------------------------------------------------------------------------

@dataclass
class SituationalAwareness:
    """Full picture of the user's current state from all sensors."""

    # Environment
    active_window: str = ""
    active_process: str = ""
    activity_type: str = "unknown"
    activity_confidence: float = 0.0

    # Cognitive state
    flow_score: float = 0.5
    in_flow: bool = False
    switch_count_30m: int = 0
    is_deep_work: bool = False

    # Emotional state
    sentiment: str = "neutral"
    arousal: str = "calm"

    # Temporal context
    current_hour: int = 0
    rhythmic_score: float = 0.5  # productivity at this hour

    # Task context
    minutes_in_task: float = 0.0
    task_description: str = ""

    # Conversation depth
    conversation_turns: int = 0

    def cognitive_load(self) -> float:
        """Estimate cognitive load: high flow + long task = high load.

        Range [0, 1]. High load means don't interrupt with complex suggestions.
        """
        flow_factor = self.flow_score
        duration_factor = min(self.minutes_in_task / 60.0, 1.0)
        deep_bonus = 0.2 if self.is_deep_work else 0.0
        return min(1.0, 0.4 * flow_factor + 0.3 * duration_factor + deep_bonus + 0.1)

    def to_context_block(self) -> str:
        """Render as a structured context block for the prompt."""
        lines = []

        if self.active_window or self.activity_type != "unknown":
            activity = self.activity_type if self.activity_type != "unknown" else "general"
            window = self.active_window or "unknown"
            lines.append(f"Activity: {activity} in {window}")

        if self.flow_score > 0.65:
            lines.append(f"Focus: Deep concentration (flow: {self.flow_score:.0%})")
        elif self.flow_score < 0.35:
            lines.append(f"Focus: Scattered (flow: {self.flow_score:.0%}, {self.switch_count_30m} switches in 30m)")

        if self.sentiment != "neutral" or self.arousal != "calm":
            lines.append(f"Mood: {self.sentiment}, {self.arousal}")

        if self.rhythmic_score > 0.7:
            lines.append(f"Energy: Peak productivity hour (score: {self.rhythmic_score:.0%})")
        elif self.rhythmic_score < 0.3:
            lines.append(f"Energy: Low productivity hour (score: {self.rhythmic_score:.0%})")

        if self.minutes_in_task > 5:
            lines.append(f"Session: {int(self.minutes_in_task)}min on current task")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cognitive Architecture
# ---------------------------------------------------------------------------

class CognitiveArchitecture:
    """Multi-stage intelligent response pipeline.

    Replaces BrainOrchestrator's dumb prompt builder with a system that
    perceives, classifies, retrieves, reasons, reflects, and adapts.
    """

    def __init__(
        self,
        model_engine,
        working_memory: WorkingMemory,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        self_reflection: Optional[SelfReflection] = None,
        system_prompt: str = "",
    ):
        self._engine = model_engine
        self._wm = working_memory
        self._em = episodic_memory
        self._sm = semantic_memory
        self._reflection = self_reflection or SelfReflection()
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Stage 1: PERCEIVE — gather situational awareness
    # ------------------------------------------------------------------

    def _perceive(self) -> SituationalAwareness:
        """Read all signals from working memory into a structured snapshot."""
        from homie_core.utils import utc_now

        sa = SituationalAwareness()
        sa.active_window = self._wm.get("active_window", "")
        sa.active_process = self._wm.get("active_process", "")
        sa.activity_type = self._wm.get("activity_type", "unknown")
        sa.activity_confidence = self._wm.get("activity_confidence", 0.0)
        sa.flow_score = self._wm.get("flow_score", 0.5)
        sa.in_flow = self._wm.get("in_flow", False)
        sa.switch_count_30m = self._wm.get("switch_count_30m", 0)
        sa.is_deep_work = self._wm.get("is_deep_work", False)
        sa.sentiment = self._wm.get("sentiment", "neutral")
        sa.arousal = self._wm.get("arousal", "calm")
        sa.rhythmic_score = self._wm.get("rhythmic_score", 0.5)
        sa.minutes_in_task = self._wm.get("minutes_in_task", 0.0)
        sa.task_description = self._wm.get("task_description", "")
        sa.current_hour = utc_now().hour
        sa.conversation_turns = len(self._wm.get_conversation())
        return sa

    # ------------------------------------------------------------------
    # Stage 2: CLASSIFY — determine query complexity
    # ------------------------------------------------------------------

    def _classify(self, query: str, awareness: SituationalAwareness) -> str:
        """Classify query complexity using text + contextual signals."""
        return classify_query_complexity(
            query, conversation_depth=awareness.conversation_turns
        )

    # ------------------------------------------------------------------
    # Stage 3: RETRIEVE — query-aware memory retrieval with TF-IDF
    # ------------------------------------------------------------------

    def _retrieve_relevant_facts(self, query: str, budget: int, max_facts: int = 10) -> list[dict]:
        """Retrieve facts ranked by TF-IDF relevance to the query."""
        if not self._sm:
            return []
        all_facts = self._sm.get_facts(min_confidence=0.4)
        if not all_facts:
            return []

        # Score by TF-IDF relevance
        fact_texts = [f["fact"] for f in all_facts]
        scores = _tf_idf_relevance(query, fact_texts)

        # Pair, sort, filter
        scored = sorted(
            zip(all_facts, scores), key=lambda x: x[1], reverse=True
        )

        # Take top facts within budget
        result = []
        used = 0
        for fact, score in scored[:max_facts]:
            text = fact["fact"]
            if used + len(text) > budget:
                break
            if score > 0.05:  # minimum relevance threshold
                result.append({**fact, "_relevance": score})
                used += len(text) + 5
        return result

    def _retrieve_relevant_episodes(self, query: str, budget: int, max_episodes: int = 3) -> list[dict]:
        """Retrieve episodes via vector search."""
        if not self._em:
            return []
        try:
            episodes = self._em.recall(query, n=max_episodes)
            # Truncate to budget
            result = []
            used = 0
            for ep in episodes:
                text = ep.get("summary", "")
                if used + len(text) > budget:
                    break
                result.append(ep)
                used += len(text) + 5
            return result
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Stage 4: REASON — build an intelligent, structured prompt
    # ------------------------------------------------------------------

    def _build_cognitive_prompt(
        self,
        query: str,
        complexity: str,
        awareness: SituationalAwareness,
        facts: list[dict],
        episodes: list[dict],
    ) -> str:
        """Build a rich, structured prompt using all available intelligence.

        The prompt is structured as cognitive context blocks, not raw text dumps.
        Budget adapts to query complexity — trivial queries get short prompts,
        deep queries get full context.
        """
        budget_cfg = _TOKEN_BUDGETS[complexity]
        max_chars = budget_cfg["prompt_chars"]

        parts: list[str] = []
        used = 0

        # 1. System prompt (always)
        parts.append(self._system_prompt)
        used += len(self._system_prompt)

        # 2. Situational awareness (except trivial)
        if complexity != QueryComplexity.TRIVIAL:
            context_block = awareness.to_context_block()
            if context_block:
                section = f"\n[CONTEXT]\n{context_block}"
                if used + len(section) < max_chars - len(query) - 100:
                    parts.append(section)
                    used += len(section)

        # 3. Relevant knowledge (moderate+ queries)
        if complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.DEEP):
            if facts:
                fact_lines = []
                for f in facts:
                    confidence = f.get("confidence", 0.5)
                    marker = "+" if confidence > 0.7 else "~"
                    fact_lines.append(f"  {marker} {f['fact']}")
                knowledge = "\n[KNOWLEDGE]\n" + "\n".join(fact_lines)
                if used + len(knowledge) < max_chars - len(query) - 100:
                    parts.append(knowledge)
                    used += len(knowledge)

        # 4. Relevant episodes (complex+ queries)
        if complexity in (QueryComplexity.COMPLEX, QueryComplexity.DEEP):
            if episodes:
                ep_lines = []
                for ep in episodes:
                    mood = ep.get("mood", "")
                    outcome = ep.get("outcome", "")
                    summary = ep.get("summary", "")
                    meta = f" ({mood}, {outcome})" if mood or outcome else ""
                    ep_lines.append(f"  - {summary}{meta}")
                memory = "\n[MEMORY]\n" + "\n".join(ep_lines)
                if used + len(memory) < max_chars - len(query) - 100:
                    parts.append(memory)
                    used += len(memory)

        # 5. Conversation history (adaptive depth)
        conversation = self._wm.get_conversation()
        if len(conversation) > 1:
            # More turns for complex queries
            max_turns = {
                QueryComplexity.TRIVIAL: 0,
                QueryComplexity.SIMPLE: 2,
                QueryComplexity.MODERATE: 4,
                QueryComplexity.COMPLEX: 8,
                QueryComplexity.DEEP: 12,
            }[complexity]

            recent = conversation[-(max_turns * 2):] if max_turns > 0 else []
            if recent:
                conv_lines = []
                for m in recent[:-1]:  # exclude current (it's in the query)
                    role = m["role"].capitalize()
                    content = m["content"]
                    # Truncate individual messages proportionally
                    max_msg = max_chars // (len(recent) + 2)
                    if len(content) > max_msg:
                        content = content[:max_msg] + "..."
                    conv_lines.append(f"  {role}: {content}")

                conv_block = "\n[CONVERSATION]\n" + "\n".join(conv_lines)
                if used + len(conv_block) < max_chars - len(query) - 50:
                    parts.append(conv_block)
                    used += len(conv_block)

        # 6. Response guidance based on user state
        guidance = self._generate_response_guidance(complexity, awareness)
        if guidance:
            parts.append(f"\n[GUIDANCE]\n{guidance}")

        # 7. User query (always last)
        parts.append(f"\nUser: {query}\nAssistant:")

        return "\n".join(parts)

    def _generate_response_guidance(
        self, complexity: str, awareness: SituationalAwareness
    ) -> str:
        """Generate response style guidance based on user's cognitive state."""
        hints = []

        # Adapt to cognitive load
        load = awareness.cognitive_load()
        if load > 0.7:
            hints.append("User is deeply focused — be brief and precise, avoid tangents.")
        elif load < 0.3:
            hints.append("User is in a relaxed state — more conversational tone is fine.")

        # Adapt to emotional state
        if awareness.arousal == "frustrated":
            hints.append("User may be frustrated — be empathetic, offer clear actionable help.")
        elif awareness.arousal == "stressed":
            hints.append("User seems stressed — prioritize the most important point first.")

        # Adapt to activity
        if awareness.activity_type == "coding":
            hints.append("User is coding — prefer code examples over explanations.")
        elif awareness.activity_type == "writing":
            hints.append("User is writing — match their tone and be concise.")
        elif awareness.activity_type == "research":
            hints.append("User is researching — provide sources and structured info.")

        # Adapt to time/energy
        if awareness.rhythmic_score < 0.3:
            hints.append("Low energy hour — keep response short and actionable.")

        return " ".join(hints) if hints else ""

    # ------------------------------------------------------------------
    # Stage 5: REFLECT — score response confidence
    # ------------------------------------------------------------------

    def _reflect_on_response(
        self, query: str, complexity: str, awareness: SituationalAwareness
    ) -> dict:
        """Use self-reflection to evaluate response strategy confidence.

        Returns config adjustments (temperature, etc.) based on confidence.
        """
        features = {
            "relevance": min(1.0, awareness.activity_confidence + 0.3),
            "helpfulness": 0.7 if complexity in (QueryComplexity.SIMPLE, QueryComplexity.MODERATE) else 0.5,
            "urgency": 0.8 if awareness.arousal in ("stressed", "frustrated") else 0.4,
        }

        result = self._reflection.score_action(
            action="respond", context={"query": query}, features=features
        )

        adjustments = {}
        if result.calibrated_confidence < 0.3:
            # Low confidence — be more conservative, hedge
            adjustments["temperature"] = 0.3
            adjustments["hedge"] = True
        elif result.calibrated_confidence > 0.8:
            # High confidence — go bold
            adjustments["temperature"] = _TOKEN_BUDGETS[complexity]["temperature"]
            adjustments["hedge"] = False

        return adjustments

    # ------------------------------------------------------------------
    # Stage 6: ADAPT — full pipeline execution
    # ------------------------------------------------------------------

    def process(self, user_input: str) -> str:
        """Full cognitive pipeline — blocking."""
        self._wm.add_message("user", user_input)

        # Stage 1: Perceive
        awareness = self._perceive()

        # Stage 2: Classify
        complexity = self._classify(user_input, awareness)

        # Stage 3: Retrieve (budget-aware)
        budget_cfg = _TOKEN_BUDGETS[complexity]
        facts = self._retrieve_relevant_facts(user_input, budget=budget_cfg["prompt_chars"] // 4)
        episodes = self._retrieve_relevant_episodes(user_input, budget=budget_cfg["prompt_chars"] // 6)

        # Stage 4: Reason
        prompt = self._build_cognitive_prompt(user_input, complexity, awareness, facts, episodes)

        # Stage 5: Reflect
        adjustments = self._reflect_on_response(user_input, complexity, awareness)
        temperature = adjustments.get("temperature", budget_cfg["temperature"])
        max_tokens = budget_cfg["max_tokens"]

        # Stage 6: Generate
        response = self._engine.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        self._wm.add_message("assistant", response)

        # Update reflection with implicit feedback (response was generated)
        self._reflection.record_feedback(temperature, True)

        return response

    def process_stream(self, user_input: str) -> Iterator[str]:
        """Full cognitive pipeline — streaming for instant first-token."""
        self._wm.add_message("user", user_input)

        # Stage 1-5 are fast (no model calls)
        awareness = self._perceive()
        complexity = self._classify(user_input, awareness)

        budget_cfg = _TOKEN_BUDGETS[complexity]
        facts = self._retrieve_relevant_facts(user_input, budget=budget_cfg["prompt_chars"] // 4)
        episodes = self._retrieve_relevant_episodes(user_input, budget=budget_cfg["prompt_chars"] // 6)

        prompt = self._build_cognitive_prompt(user_input, complexity, awareness, facts, episodes)
        adjustments = self._reflect_on_response(user_input, complexity, awareness)
        temperature = adjustments.get("temperature", budget_cfg["temperature"])
        max_tokens = budget_cfg["max_tokens"]

        # Stage 6: Stream
        chunks = []
        for token in self._engine.stream(prompt, max_tokens=max_tokens, temperature=temperature):
            chunks.append(token)
            yield token

        full_response = "".join(chunks)
        self._wm.add_message("assistant", full_response)
        self._reflection.record_feedback(temperature, True)

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def get_query_analysis(self, query: str) -> dict:
        """Expose the cognitive analysis for debugging/transparency."""
        awareness = self._perceive()
        complexity = self._classify(query, awareness)
        budget = _TOKEN_BUDGETS[complexity]
        facts = self._retrieve_relevant_facts(query, budget=budget["prompt_chars"] // 4)
        episodes = self._retrieve_relevant_episodes(query, budget=budget["prompt_chars"] // 6)
        return {
            "complexity": complexity,
            "budget": budget,
            "n_relevant_facts": len(facts),
            "n_relevant_episodes": len(episodes),
            "awareness": {
                "activity": awareness.activity_type,
                "flow": awareness.flow_score,
                "sentiment": awareness.sentiment,
                "arousal": awareness.arousal,
                "cognitive_load": awareness.cognitive_load(),
                "rhythmic_score": awareness.rhythmic_score,
            },
        }
