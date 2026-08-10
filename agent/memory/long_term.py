"""Long-term memory with SQLite persistence."""

import sqlite3
import time
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import MemoryRecord, MemoryType


class LongTermMemory:
    """SQLite-backed long-term memory storage."""

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    source TEXT DEFAULT 'conversation',
                    timestamp REAL NOT NULL,
                    importance REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 0.8,
                    session_id TEXT,
                    metadata TEXT DEFAULT '{}',
                    active INTEGER DEFAULT 1
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_active ON memories(active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_importance ON memories(importance DESC)")
            conn.commit()
        finally:
            conn.close()

    def store(self, record: MemoryRecord) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (memory_id, content, memory_type, source, timestamp,
                 importance, confidence, session_id, metadata, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.memory_id,
                record.content,
                record.memory_type.value,
                record.source,
                record.timestamp,
                record.importance,
                record.confidence,
                record.session_id,
                json.dumps(record.metadata),
                1 if record.active else 0,
            ))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def retrieve(self, query: str = "", limit: int = 10,
                 memory_type: Optional[MemoryType] = None,
                 min_importance: float = 0.0) -> List[MemoryRecord]:
        conn = self._get_conn()
        try:
            sql = "SELECT * FROM memories WHERE active = 1"
            params: list = []

            if query:
                sql += " AND content LIKE ?"
                params.append(f"%{query}%")

            if memory_type:
                sql += " AND memory_type = ?"
                params.append(memory_type.value)

            if min_importance > 0:
                sql += " AND importance >= ?"
                params.append(min_importance)

            sql += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            conn.close()

    def get_all(self, limit: int = 100) -> List[MemoryRecord]:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE active = 1 "
                "ORDER BY importance DESC, timestamp DESC LIMIT ?",
                (limit,)
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete(self, memory_id: str) -> bool:
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE memories SET active = 0 WHERE memory_id = ?",
                (memory_id,)
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def hard_delete(self, memory_id: str) -> bool:
        conn = self._get_conn()
        try:
            conn.execute(
                "DELETE FROM memories WHERE memory_id = ?",
                (memory_id,)
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def update(self, record: MemoryRecord) -> bool:
        return self.store(record)

    def get_count(self) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE active = 1"
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def get_type_counts(self) -> Dict[str, int]:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT memory_type, COUNT(*) FROM memories "
                "WHERE active = 1 GROUP BY memory_type"
            )
            return {row[0]: row[1] for row in cursor.fetchall()}
        finally:
            conn.close()

    def clear(self) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM memories")
            conn.commit()
            return True
        finally:
            conn.close()

    def health_check(self) -> Dict[str, Any]:
        count = self.get_count()
        type_counts = self.get_type_counts()
        return {
            "status": "healthy",
            "total_memories": count,
            "type_breakdown": type_counts,
            "db_path": str(self.db_path),
            "db_size_bytes": os.path.getsize(self.db_path) if self.db_path.exists() else 0,
        }

    def _row_to_record(self, row: tuple) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row[0],
            content=row[1],
            memory_type=MemoryType(row[2]),
            source=row[3],
            timestamp=row[4],
            importance=row[5],
            confidence=row[6],
            session_id=row[7],
            metadata=json.loads(row[8]) if row[8] else {},
            active=bool(row[9]),
        )
