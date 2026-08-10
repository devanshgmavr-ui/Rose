"""Context window management for LLM interactions."""

import time
from typing import List, Optional, Dict, Any

from .base import Message, MessageRole, MemoryRecord, MemoryType
from .conversation import ConversationManager
from .long_term import LongTermMemory


class ContextManager:
    """Manages context window budget for LLM interactions."""

    def __init__(
        self,
        conversation_manager: ConversationManager,
        long_term_memory: LongTermMemory,
        system_prompt: str = "",
        max_context_tokens: int = 3500,
        reserved_output_tokens: int = 512,
        recent_message_count: int = 6,
        memory_retrieval_limit: int = 5,
    ):
        self.conversation_manager = conversation_manager
        self.long_term_memory = long_term_memory
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.recent_message_count = recent_message_count
        self.memory_retrieval_limit = memory_retrieval_limit

    def build_context(
        self,
        user_query: str,
        include_memories: bool = True,
        session_id: Optional[str] = None,
        tool_results: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build optimized context messages for LLM."""
        context = []
        used_tokens = 0

        available_tokens = self.max_context_tokens - self.reserved_output_tokens

        if self.system_prompt:
            system_tokens = self._estimate_tokens(self.system_prompt)
            if system_tokens < available_tokens * 0.3:
                context.append({"role": "system", "content": self.system_prompt})
                used_tokens += system_tokens

        if include_memories:
            memories = self._retrieve_relevant_memories(user_query, session_id)
            memory_text = self._format_memories_for_context(memories)
            if memory_text:
                memory_tokens = self._estimate_tokens(memory_text)
                if used_tokens + memory_tokens < available_tokens * 0.5:
                    context.append({
                        "role": "system",
                        "content": f"Relevant context from memory:\n{memory_text}"
                    })
                    used_tokens += memory_tokens

        if tool_results:
            for tr in tool_results:
                tr_tokens = self._estimate_tokens(tr.get("content", ""))
                if used_tokens + tr_tokens < available_tokens * 0.7:
                    context.append(tr)
                    used_tokens += tr_tokens

        remaining_tokens = available_tokens - used_tokens
        recent_msgs = self.conversation_manager.get_messages(limit=self.recent_message_count)

        if not recent_msgs and user_query:
            context.append({"role": "user", "content": user_query})
            return context

        selected_msgs = []
        token_budget = remaining_tokens

        for msg in reversed(recent_msgs):
            msg_tokens = self._estimate_tokens(msg.content)
            if msg_tokens <= token_budget:
                selected_msgs.insert(0, msg)
                token_budget -= msg_tokens
            else:
                break

        if user_query:
            query_tokens = self._estimate_tokens(user_query)
            if query_tokens <= token_budget:
                context.append({"role": "user", "content": user_query})
                used_tokens += query_tokens

        for msg in selected_msgs:
            context.append({
                "role": msg.role.value,
                "content": msg.content
            })

        return context

    def _retrieve_relevant_memories(
        self, query: str, session_id: Optional[str] = None
    ) -> List[MemoryRecord]:
        memories = self.long_term_memory.retrieve(
            query=query,
            limit=self.memory_retrieval_limit,
        )

        existing_ids = {m.memory_id for m in memories}

        if session_id:
            session_memories = self.long_term_memory.retrieve(
                query="",
                limit=3,
                min_importance=0.6,
            )
            for sm in session_memories:
                if sm.memory_id not in existing_ids and sm.session_id == session_id:
                    memories.append(sm)
                    existing_ids.add(sm.memory_id)

        if len(memories) < self.memory_retrieval_limit:
            additional = self.long_term_memory.retrieve(
                query="",
                limit=self.memory_retrieval_limit - len(memories),
                min_importance=0.6,
            )
            for mem in additional:
                if mem.memory_id not in existing_ids:
                    memories.append(mem)
                    existing_ids.add(mem.memory_id)

        memories.sort(key=lambda m: m.importance, reverse=True)
        return memories[:self.memory_retrieval_limit]

    def _format_memories_for_context(self, memories: List[MemoryRecord]) -> str:
        if not memories:
            return ""

        lines = []
        for mem in memories:
            type_label = mem.memory_type.value.replace("_", " ").title()
            confidence_pct = int(mem.confidence * 100)
            lines.append(f"- [{type_label}] {mem.content} (confidence: {confidence_pct}%)")

        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def get_context_stats(self) -> Dict[str, Any]:
        system_tokens = self._estimate_tokens(self.system_prompt) if self.system_prompt else 0
        all_messages = self.conversation_manager.get_messages()
        recent_messages = self.conversation_manager.get_messages(limit=self.recent_message_count)
        msg_tokens = sum(self._estimate_tokens(m.content) for m in recent_messages)
        total_tokens = sum(self._estimate_tokens(m.content) for m in all_messages)

        return {
            "system_prompt_tokens": system_tokens,
            "recent_messages_tokens": msg_tokens,
            "total_conversation_tokens": total_tokens,
            "available_tokens": self.max_context_tokens - self.reserved_output_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "estimated_usage": system_tokens + total_tokens,
            "estimated_remaining": (
                self.max_context_tokens
                - self.reserved_output_tokens
                - system_tokens
                - total_tokens
            ),
        }

    def needs_summarization(self) -> bool:
        stats = self.get_context_stats()
        return stats["estimated_remaining"] < 200

    def update_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def update_limits(self, max_tokens: int, reserved_output: int):
        self.max_context_tokens = max_tokens
        self.reserved_output_tokens = reserved_output
