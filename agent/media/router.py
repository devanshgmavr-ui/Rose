"""Media router for dispatching requests to appropriate providers."""

import logging
import time
from typing import Optional, Dict, List, Tuple

from .base import (
    MediaProvider, MediaType, MediaRequest, MediaResult, MediaOutput
)
from .storage import MediaStorage

logger = logging.getLogger(__name__)


class MediaRouter:
    def __init__(self, storage: Optional[MediaStorage] = None):
        self._providers: Dict[str, MediaProvider] = {}
        self._type_providers: Dict[MediaType, List[str]] = {
            MediaType.IMAGE: [],
            MediaType.VIDEO: [],
        }
        self._storage = storage or MediaStorage()
        self._request_count = 0
        self._total_time = 0.0

    @property
    def storage(self) -> MediaStorage:
        return self._storage

    def register_provider(self, provider: MediaProvider) -> bool:
        if provider.name in self._providers:
            logger.warning(f"Provider already registered: {provider.name}")
            return False

        self._providers[provider.name] = provider

        media_type = provider.media_type
        if media_type not in self._type_providers:
            self._type_providers[media_type] = []
        self._type_providers[media_type].append(provider.name)

        logger.info(f"Registered media provider: {provider.name} ({media_type.value})")
        return True

    def unregister_provider(self, name: str) -> bool:
        if name not in self._providers:
            return False

        provider = self._providers.pop(name)
        type_list = self._type_providers.get(provider.media_type, [])
        if name in type_list:
            type_list.remove(name)

        logger.info(f"Unregistered media provider: {name}")
        return True

    def get_provider(self, name: str) -> Optional[MediaProvider]:
        return self._providers.get(name)

    def get_providers_for_type(self, media_type: MediaType) -> List[MediaProvider]:
        names = self._type_providers.get(media_type, [])
        return [self._providers[n] for n in names if n in self._providers]

    def list_providers(self) -> List[Dict]:
        return [p.get_info() for p in self._providers.values()]

    def list_provider_names(self) -> List[str]:
        return list(self._providers.keys())

    def select_provider(
        self,
        media_type: MediaType,
        preferred: Optional[str] = None,
    ) -> Optional[MediaProvider]:
        if preferred:
            provider = self._providers.get(preferred)
            if provider and provider.media_type == media_type:
                return provider

        available = [
            p for p in self.get_providers_for_type(media_type)
            if p.is_available
        ]

        if not available:
            return None

        return available[0]

    def route(
        self,
        request: MediaRequest,
        provider_name: Optional[str] = None,
    ) -> MediaResult:
        start = time.time()
        self._request_count += 1

        provider = self.select_provider(request.media_type, provider_name)
        if not provider:
            return MediaResult(
                success=False,
                media_type=request.media_type,
                error=f"No provider available for {request.media_type.value}",
                provider=provider_name or "none",
            )

        if not provider.is_available:
            return MediaResult(
                success=False,
                media_type=request.media_type,
                error=f"Provider {provider.name} is not available",
                provider=provider.name,
            )

        valid, errors = provider.validate_request(request)
        if not valid:
            return MediaResult(
                success=False,
                media_type=request.media_type,
                error="Validation failed: " + "; ".join(errors),
                provider=provider.name,
            )

        result = provider.process(request)
        result.provider = provider.name

        elapsed = time.time() - start
        self._total_time += elapsed

        if result.success and result.output:
            self._store_result(result, request)

        return result

    def _store_result(self, result: MediaResult, request: MediaRequest) -> None:
        if not result.output:
            return

        output_path = result.output.path
        if not output_path:
            return

        from pathlib import Path
        p = Path(output_path)
        if not p.exists():
            return

        if not self._storage._is_within_workspace(p.resolve()):
            logger.warning(f"Generated file outside workspace: {output_path}")
            return

        result.metadata["stored_path"] = output_path

    def get_stats(self) -> Dict:
        return {
            "provider_count": len(self._providers),
            "providers": self.list_provider_names(),
            "total_requests": self._request_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / max(self._request_count, 1),
            "storage_stats": self._storage.get_storage_stats(),
        }
