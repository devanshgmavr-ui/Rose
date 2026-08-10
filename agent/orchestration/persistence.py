"""Task persistence for orchestration."""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from .models import Task


class TaskPersistence:
    """Persists task state to JSON files."""

    def __init__(self, data_dir: str = "data/tasks"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def save_task(self, task: Task) -> bool:
        try:
            path = self._data_dir / f"{task.task_id}.json"
            data = task.to_dict()
            data["updated_at"] = time.time()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def load_task(self, task_id: str) -> Optional[Task]:
        path = self._data_dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Task.from_dict(data)
        except Exception:
            return None

    def list_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        tasks = []
        for path in sorted(self._data_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tasks.append({
                    "task_id": data.get("task_id"),
                    "user_request": data.get("user_request", "")[:100],
                    "status": data.get("status"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                })
                if len(tasks) >= limit:
                    break
            except Exception:
                continue
        return tasks

    def delete_task(self, task_id: str) -> bool:
        path = self._data_dir / f"{task_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def get_task_count(self) -> int:
        return len(list(self._data_dir.glob("*.json")))
