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
from homie_core.brain.tool_registry import ToolRegistry, parse_tool_calls
from homie_core.middleware.hooks import HookRegistry, PipelineStage, RetrievalBundle
from homie_core.brain.agentic_loop import AgenticLoop, _strip_tool_markers
from homie_core.brain.persona import select_persona, get_persona_guidance
from homie_core.brain.iteration_budget import IterationBudget
from homie_core.brain.context_compressor import ContextCompressor
from homie_core.memory.learning_pipeline import LearningPipeline
from homie_core.memory.user_model import UserModelSynthesizer
from homie_core.security.injection_detector import sanitize_external_content
from homie_core.security.redact import redact_sensitive_text
from homie_core.rag.pipeline import RagPipeline


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
    QueryComplexity.TRIVIAL:  {"max_tokens": 256,  "prompt_chars": 1500,  "temperature": 0.8},
    QueryComplexity.SIMPLE:   {"max_tokens": 384,  "prompt_chars": 2500,  "temperature": 0.7},
    QueryComplexity.MODERATE: {"max_tokens": 768,  "prompt_chars": 5000,  "temperature": 0.7},
    QueryComplexity.COMPLEX:  {"max_tokens": 1536, "prompt_chars": 8000,  "temperature": 0.6},
    QueryComplexity.DEEP:     {"max_tokens": 3072, "prompt_chars": 12000, "temperature": 0.5},
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
        tool_registry: Optional[ToolRegistry] = None,
        rag_pipeline: Optional[RagPipeline] = None,
        hooks: Optional[HookRegistry] = None,
    ):
        self._engine = model_engine
        self._wm = working_memory
        self._em = episodic_memory
        self._sm = semantic_memory
        self._reflection = self_reflection or SelfReflection()
        self._system_prompt = system_prompt
        self._tools = tool_registry
        self._rag = rag_pipeline
        self._hooks = hooks or HookRegistry()
        self._budget = IterationBudget(max_iterations=20)
        self._agentic = AgenticLoop(model_engine, tool_registry, budget=self._budget) if tool_registry else None
        self._compressor = ContextCompressor()
        self._learning = LearningPipeline(
            semantic_memory=semantic_memory,
            episodic_memory=episodic_memory,
        )
        self._user_model = UserModelSynthesizer(
            semantic_memory=semantic_memory,
            episodic_memory=episodic_memory,
        )
        # Conversation meta-tracking
        self._topic_history: list[str] = []
        self._user_engagement: float = 0.5  # 0=disengaged, 1=highly engaged

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

    def _retrieve_documents(self, query: str, budget: int, top_k: int = 5) -> str:
        """Retrieve relevant document chunks via RAG pipeline.

        Returns a formatted [DOCUMENTS] block for prompt injection.
        """
        if not self._rag:
            return ""
        try:
            return self._rag.build_context_block(query, max_chars=budget, top_k=top_k)
        except Exception:
            return ""

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
        documents_block: str = "",
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

        # 3. Relevant knowledge (simple+ queries — facts help even for simple questions)
        if complexity in (QueryComplexity.SIMPLE, QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.DEEP):
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

        # 5. Retrieved documents (moderate+ queries with RAG)
        if complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.DEEP):
            if documents_block:
                if used + len(documents_block) < max_chars - len(query) - 100:
                    parts.append(f"\n{documents_block}")
                    used += len(documents_block)

        # 6. Conversation history (smart compression)
        conversation = self._wm.get_conversation()
        if len(conversation) > 1:
            # More turns for complex queries
            max_turns = {
                QueryComplexity.TRIVIAL: 1,
                QueryComplexity.SIMPLE: 3,
                QueryComplexity.MODERATE: 6,
                QueryComplexity.COMPLEX: 10,
                QueryComplexity.DEEP: 16,
            }[complexity]

            all_msgs = conversation[:-1]  # exclude current (it's in the query)

            # Smart compression: keep recent turns verbatim, summarize older ones
            recent_count = min(max_turns, len(all_msgs))
            recent = all_msgs[-recent_count:] if recent_count > 0 else []
            older = all_msgs[:-recent_count] if recent_count < len(all_msgs) else []

            conv_lines = []

            # Summarize older messages if they exist and query is complex enough
            if older and complexity in (QueryComplexity.COMPLEX, QueryComplexity.DEEP):
                user_topics = []
                for m in older:
                    if m["role"] == "user":
                        # Extract key nouns/verbs from older messages
                        words = _tokenize(m["content"])
                        key_words = [w for w in words if len(w) > 3][:5]
                        if key_words:
                            user_topics.extend(key_words)
                if user_topics:
                    # Deduplicate while preserving order
                    seen = set()
                    unique_topics = []
                    for t in user_topics:
                        if t not in seen:
                            seen.add(t)
                            unique_topics.append(t)
                    summary = f"  [Earlier: discussed {', '.join(unique_topics[:8])}]"
                    conv_lines.append(summary)

            # Add recent messages verbatim
            for m in recent:
                role = m["role"].capitalize()
                content = m["content"]
                # Truncate individual messages proportionally
                max_msg = max_chars // (len(recent) + 4)
                if len(content) > max_msg:
                    content = content[:max_msg] + "..."
                conv_lines.append(f"  {role}: {content}")

            if conv_lines:
                conv_block = "\n[CONVERSATION]\n" + "\n".join(conv_lines)
                if used + len(conv_block) < max_chars - len(query) - 50:
                    parts.append(conv_block)
                    used += len(conv_block)

        # 7. Topic continuity (helps maintain coherent multi-turn conversations)
        if self._topic_history and complexity != QueryComplexity.TRIVIAL:
            recent_topics = self._topic_history[-5:]
            if recent_topics:
                topic_str = " → ".join(recent_topics)
                topic_section = f"\n[TOPIC FLOW]\n{topic_str}"
                if used + len(topic_section) < max_chars - len(query) - 50:
                    parts.append(topic_section)
                    used += len(topic_section)

        # 8. User profile (moderate+ queries)
        if complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.DEEP):
            user_context = self._user_model.get_relevant_context(query, max_chars=400)
            if user_context:
                section = f"\n[USER PROFILE]\n{user_context}"
                if used + len(section) < max_chars - len(query) - 50:
                    parts.append(section)
                    used += len(section)

        # 9. Project context (if active project detected)
        if self._learning.active_project and complexity != QueryComplexity.TRIVIAL:
            project = self._learning.active_project
            project_section = f"\n[PROJECT]\nActive project: {project}"
            if used + len(project_section) < max_chars - len(query) - 50:
                parts.append(project_section)
                used += len(project_section)

        # 10. Response guidance based on user state + persona
        guidance = self._generate_response_guidance(complexity, awareness)
        if guidance:
            parts.append(f"\n[GUIDANCE]\n{guidance}")

        # 9. User query (always last)
        parts.append(f"\nUser: {query}\nAssistant:")

        return "\n".join(parts)

    def _generate_response_guidance(
        self, complexity: str, awareness: SituationalAwareness
    ) -> str:
        """Generate response style guidance based on user's cognitive state and persona."""
        hints = []

        # Specialist persona guidance (replaces simple activity hints)
        persona_hint = get_persona_guidance(
            awareness.activity_type,
            awareness.activity_confidence,
        )
        if persona_hint:
            hints.append(persona_hint)

        # Chain-of-thought for complex queries
        if complexity in (QueryComplexity.COMPLEX, QueryComplexity.DEEP):
            hints.append(
                "This is a complex question. Think step by step: "
                "1) Identify what's being asked, 2) Consider the key factors, "
                "3) Reason through the options, 4) Give a clear recommendation. "
                "Show your reasoning briefly before the conclusion."
            )

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

        # Adapt to time/energy
        if awareness.rhythmic_score < 0.3:
            hints.append("Low energy hour — keep response short and actionable.")

        # Conversation continuity
        if awareness.conversation_turns > 6:
            hints.append(
                "This is a long conversation — reference earlier context naturally. "
                "Don't repeat what was already discussed."
            )

        # Engagement-based adaptation
        if hasattr(self, '_user_engagement'):
            if self._user_engagement > 0.7:
                hints.append("User is highly engaged — give thorough, detailed responses.")
            elif self._user_engagement < 0.3:
                hints.append("User seems disengaged — be more concise and direct. Maybe ask if they need something different.")

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

    def _prepare_prompt(self, user_input: str):
        """Run stages 1-5 and return (prompt, budget_cfg, adjustments).

        Enhanced with:
        - Context compression for long conversations
        - Injection detection on external context (RAG documents)
        - Secret redaction on tool-generated content
        """
        # Auto-compress conversation if it's getting too long
        conversation = self._wm.get_conversation()
        if self._compressor.needs_compression(conversation):
            compressed = self._compressor.compress(conversation)
            self._wm._conversation = compressed  # Replace in-place

        awareness = self._perceive()
        awareness = self._hooks.emit(PipelineStage.PERCEIVED, awareness)
        complexity = self._classify(user_input, awareness)
        complexity = self._hooks.emit(PipelineStage.CLASSIFIED, complexity)

        budget_cfg = _TOKEN_BUDGETS[complexity]
        facts = self._retrieve_relevant_facts(user_input, budget=budget_cfg["prompt_chars"] // 4)
        episodes = self._retrieve_relevant_episodes(user_input, budget=budget_cfg["prompt_chars"] // 6)

        # RAG: retrieve relevant documents for moderate+ queries
        # Apply injection detection on retrieved documents
        documents_block = ""
        if complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.DEEP):
            raw_docs = self._retrieve_documents(user_input, budget=budget_cfg["prompt_chars"] // 3)
            if raw_docs:
                safe_docs, scan_result = sanitize_external_content(raw_docs)
                if scan_result.threat_level in ("high", "critical"):
                    # Block suspicious documents entirely
                    documents_block = ""
                else:
                    documents_block = safe_docs

        bundle = RetrievalBundle(facts=facts, episodes=episodes, documents=documents_block)
        bundle = self._hooks.emit(PipelineStage.RETRIEVED, bundle)
        facts = bundle.facts
        episodes = bundle.episodes
        documents_block = bundle.documents

        prompt = self._build_cognitive_prompt(user_input, complexity, awareness, facts, episodes, documents_block)

        # Inject tool descriptions if tools are available
        if self._tools:
            tool_prompt = self._tools.generate_tool_prompt()
            if tool_prompt:
                prompt = prompt.replace("\nUser:", f"\n{tool_prompt}\n\nUser:")

        prompt = self._hooks.emit(PipelineStage.PROMPT_BUILT, prompt)

        adjustments = self._reflect_on_response(user_input, complexity, awareness)
        adjustments = self._hooks.emit(PipelineStage.REFLECTED, adjustments)
        temperature = adjustments.get("temperature", budget_cfg["temperature"])

        return prompt, budget_cfg, temperature

    def _track_conversation_meta(self, user_input: str) -> None:
        """Track conversation-level metadata: topics, engagement, mood shifts."""
        # Extract topic keywords from current message
        words = _tokenize(user_input)
        stop = {"the", "a", "an", "is", "are", "was", "were", "be", "have", "has",
                "do", "does", "did", "will", "would", "could", "should", "can",
                "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
                "and", "but", "or", "not", "no", "so", "if", "it", "my", "me",
                "you", "your", "we", "they", "what", "how", "when", "where", "why",
                "this", "that", "i", "just", "about"}
        topic_words = [w for w in words if w not in stop and len(w) > 2]
        if topic_words:
            self._topic_history.append(" ".join(topic_words[:5]))
            # Keep only recent topics
            if len(self._topic_history) > 20:
                self._topic_history = self._topic_history[-20:]

        # Estimate engagement from message length and question marks
        length_signal = min(len(user_input) / 200.0, 1.0)
        question_signal = 0.3 if "?" in user_input else 0.0
        detail_signal = 0.2 if len(words) > 15 else 0.0
        # Exponential moving average
        new_engagement = length_signal + question_signal + detail_signal
        self._user_engagement = 0.7 * self._user_engagement + 0.3 * min(1.0, new_engagement)

    def process(self, user_input: str) -> str:
        """Full cognitive pipeline — blocking, with agentic loop + learning."""
        self._wm.add_message("user", user_input)

        # Learn from user input (lightweight pattern extraction)
        self._learning.process_user_message(user_input)
        self._track_conversation_meta(user_input)

        # Stages 1-5: Perceive, Classify, Retrieve, Reason, Reflect
        prompt, budget_cfg, temperature = self._prepare_prompt(user_input)
        max_tokens = budget_cfg["max_tokens"]

        # Stage 6: Generate (with agentic loop if tools available)
        if self._agentic:
            response = self._agentic.process(prompt, max_tokens=max_tokens, temperature=temperature)
        else:
            response = self._engine.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        self._wm.add_message("assistant", response)
        self._reflection.record_feedback(temperature, True)

        return response

    def process_stream(self, user_input: str) -> Iterator[str]:
        """Full cognitive pipeline — streaming, with learning."""
        self._wm.add_message("user", user_input)

        # Learn from user input
        self._learning.process_user_message(user_input)
        self._track_conversation_meta(user_input)

        # Stages 1-5
        prompt, budget_cfg, temperature = self._prepare_prompt(user_input)
        max_tokens = budget_cfg["max_tokens"]

        # Stage 6: Stream (with agentic loop if tools available)
        chunks = []
        if self._agentic:
            for token in self._agentic.process_stream(prompt, max_tokens=max_tokens, temperature=temperature):
                chunks.append(token)
                yield token
        else:
            for token in self._engine.stream(prompt, max_tokens=max_tokens, temperature=temperature):
                chunks.append(token)
                yield token

        full_response = "".join(chunks)
        self._wm.add_message("assistant", full_response)
        self._reflection.record_feedback(temperature, True)

    def consolidate_session(self, mood: Optional[str] = None) -> Optional[str]:
        """Consolidate current session into episodic memory. Call at session end."""
        return self._learning.consolidate_session(self._wm, mood=mood)

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
