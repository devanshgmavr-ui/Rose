"""Conversation summarization for context management."""

import time
from typing import List, Optional, Dict, Any, Tuple

from .base import Message, MessageRole, MemoryRecord, MemoryType
from .conversation import ConversationManager
from .long_term import LongTermMemory


class ConversationSummarizer:
    """Summarizes conversations to preserve context within token limits."""

    def __init__(
        self,
        conversation_manager: ConversationManager,
        long_term_memory: LongTermMemory,
        max_context_tokens: int = 3500,
        summary_threshold: int = 2500,
        preserve_recent: int = 6,
    ):
        self.conversation_manager = conversation_manager
        self.long_term_memory = long_term_memory
        self.max_context_tokens = max_context_tokens
        self.summary_threshold = summary_threshold
        self.preserve_recent = preserve_recent

    def should_summarize(self) -> bool:
        stats = self.conversation_manager.get_conversation_stats()
        token_est = stats.get("token_estimate", 0)
        return token_est > self.summary_threshold

    def summarize_and_compress(self) -> Optional[str]:
        if not self.conversation_manager.current_session:
            return None

        session = self.conversation_manager.current_session
        messages = session.messages

        if len(messages) <= self.preserve_recent:
            return None

        old_messages = messages[:-self.preserve_recent]
        recent_messages = messages[-self.preserve_recent:]

        summary = self._generate_summary(old_messages)

        self._extract_and_store_memories(old_messages, session.session_id)

        session.messages = recent_messages
        session.summary = summary
        session.updated_at = time.time()

        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[Previous conversation summary]\n{summary}",
            metadata={"type": "summary", "timestamp": time.time()},
        )
        session.messages.insert(0, summary_msg)

        self.conversation_manager.session_manager.save_session(session)

        return summary

    def _generate_summary(self, messages: List[Message]) -> str:
        topics = []
        key_points = []
        user_preferences = []
        decisions = []

        for msg in messages:
            content_lower = msg.content.lower()

            if msg.role == MessageRole.USER:
                if any(kw in content_lower for kw in ["prefer", "like", "want", "need", "use"]):
                    user_preferences.append(msg.content[:100])

            if any(kw in content_lower for kw in ["decide", "choose", "go with", "use"]):
                decisions.append(msg.content[:100])

            words = msg.content.split()
            if len(words) > 5:
                topics.extend(words[:3])

        topic_freq = {}
        for word in topics:
            w = word.lower().strip(".,!?")
            if len(w) > 3:
                topic_freq[w] = topic_freq.get(w, 0) + 1

        main_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        summary_parts = []
        if main_topics:
            summary_parts.append(
                "Main topics: " + ", ".join(t[0] for t in main_topics)
            )
        if user_preferences:
            summary_parts.append(
                "User preferences: " + "; ".join(user_preferences[:3])
            )
        if decisions:
            summary_parts.append(
                "Key decisions: " + "; ".join(decisions[:3])
            )

        if not summary_parts:
            summary_parts.append(
                f"Conversation covered {len(messages)} messages "
                f"from {time.strftime('%Y-%m-%d %H:%M', time.localtime(messages[0].timestamp))} "
                f"to {time.strftime('%Y-%m-%d %H:%M', time.localtime(messages[-1].timestamp))}"
            )

        return "\n".join(summary_parts)

    def _extract_and_store_memories(
        self, messages: List[Message], session_id: str
    ) -> List[MemoryRecord]:
        stored = []
        for msg in messages:
            memory_type, importance = self._classify_message(msg)
            if importance >= 0.5:
                record = MemoryRecord(
                    content=msg.content[:500],
                    memory_type=memory_type,
                    source="conversation_summarizer",
                    timestamp=msg.timestamp,
                    importance=importance,
                    confidence=0.7,
                    session_id=session_id,
                )
                if self.long_term_memory.store(record):
                    stored.append(record)
        return stored

    def _classify_message(self, msg: Message) -> Tuple[MemoryType, float]:
        content = msg.content.lower()

        if msg.role != MessageRole.USER:
            return MemoryType.INTERACTION, 0.3

        if any(kw in content for kw in ["prefer", "like", "always", "never", "use"]):
            return MemoryType.USER_PREFERENCE, 0.8

        if any(kw in content for kw in ["decide", "choose", "go with"]):
            return MemoryType.DECISION, 0.7

        if any(kw in content for kw in ["is a", "are the", "means", "defined as"]):
            return MemoryType.FACT, 0.7

        if any(kw in content for kw in ["project", "repo", "codebase", "file"]):
            return MemoryType.PROJECT_INFO, 0.6

        if any(kw in content for kw in ["remember", "note", "important"]):
            return MemoryType.INSTRUCTION, 0.9

        return MemoryType.INTERACTION, 0.3

    def get_summarization_stats(self) -> Dict[str, Any]:
        stats = self.conversation_manager.get_conversation_stats()
        return {
            "message_count": stats["message_count"],
            "token_estimate": stats["token_estimate"],
            "should_summarize": self.should_summarize(),
            "summary_threshold": self.summary_threshold,
            "preserve_recent": self.preserve_recent,
            "has_existing_summary": (
                self.conversation_manager.current_session is not None
                and self.conversation_manager.current_session.summary is not None
            ),
        }

    def update_limits(self, max_tokens: int, threshold: int, preserve: int):
        self.max_context_tokens = max_tokens
        self.summary_threshold = threshold
        self.preserve_recent = preserve
