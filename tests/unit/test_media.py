"""Unit tests for the multimodal media system."""

import os
import tempfile
import pytest
from pathlib import Path

from agent.media.base import (
    MediaType, MediaFormat, MediaInput, MediaOutput,
    MediaRequest, MediaResult, MediaProvider,
    MEDIA_TYPE_EXTENSIONS, MEDIA_MIME_TYPES,
)
from agent.media.storage import MediaStorage
from agent.media.router import MediaRouter
from agent.media.vision import VisionProvider, StubLocalVisionProvider
from agent.media.image_gen import ImageGenProvider, StubLocalImageGenProvider
from agent.media.video_gen import VideoGenProvider, StubLocalVideoGenProvider
from agent.media.tools import (
    ImageAnalyzeTool, ImageGenerateTool,
    VideoGenerateTool, MediaInfoTool,
)


class TestMediaEnums:
    def test_media_type_values(self):
        assert MediaType.IMAGE.value == "image"
        assert MediaType.VIDEO.value == "video"
        assert MediaType.AUDIO.value == "audio"

    def test_media_format_values(self):
        assert MediaFormat.PNG.value == "png"
        assert MediaFormat.MP4.value == "mp4"

    def test_media_type_extensions(self):
        assert ".png" in MEDIA_TYPE_EXTENSIONS[MediaType.IMAGE]
        assert ".mp4" in MEDIA_TYPE_EXTENSIONS[MediaType.VIDEO]

    def test_media_mime_types(self):
        assert MEDIA_MIME_TYPES["image/png"] == MediaType.IMAGE
        assert MEDIA_MIME_TYPES["video/mp4"] == MediaType.VIDEO


class TestMediaInput:
    def test_creation(self):
        mi = MediaInput(media_type=MediaType.IMAGE, prompt="test")
        assert mi.media_type == MediaType.IMAGE
        assert mi.prompt == "test"

    def test_to_dict(self):
        mi = MediaInput(media_type=MediaType.IMAGE, prompt="test")
        d = mi.to_dict()
        assert d["media_type"] == "image"
        assert d["prompt"] == "test"
        assert d["has_data"] is False


class TestMediaOutput:
    def test_creation(self):
        mo = MediaOutput(media_type=MediaType.IMAGE, path="/test.png")
        assert mo.media_type == MediaType.IMAGE
        assert mo.path == "/test.png"

    def test_to_dict_and_from_dict(self):
        mo = MediaOutput(
            media_type=MediaType.IMAGE, path="/test.png",
            width=512, height=512, file_size=1024,
        )
        d = mo.to_dict()
        mo2 = MediaOutput.from_dict(d)
        assert mo2.media_type == MediaType.IMAGE
        assert mo2.path == "/test.png"
        assert mo2.width == 512

    def test_defaults(self):
        mo = MediaOutput(media_type=MediaType.VIDEO, path="/test.mp4")
        assert mo.width == 0
        assert mo.height == 0
        assert mo.duration == 0.0


class TestMediaRequest:
    def test_creation(self):
        mr = MediaRequest(media_type=MediaType.IMAGE, prompt="a cat")
        assert mr.media_type == MediaType.IMAGE
        assert mr.prompt == "a cat"
        assert len(mr.request_id) == 8

    def test_to_dict_and_from_dict(self):
        mr = MediaRequest(
            media_type=MediaType.IMAGE, prompt="test",
            width=512, height=512, seed=42,
        )
        d = mr.to_dict()
        mr2 = MediaRequest.from_dict(d)
        assert mr2.prompt == "test"
        assert mr2.width == 512
        assert mr2.seed == 42


class TestMediaResult:
    def test_success(self):
        mr = MediaResult(success=True, media_type=MediaType.IMAGE)
        assert mr.success is True
        assert mr.output is None

    def test_with_output(self):
        mo = MediaOutput(media_type=MediaType.IMAGE, path="/test.png")
        mr = MediaResult(success=True, media_type=MediaType.IMAGE, output=mo)
        d = mr.to_dict()
        assert d["success"] is True
        assert "output" in d

    def test_error(self):
        mr = MediaResult(success=False, error="failed")
        d = mr.to_dict()
        assert d["success"] is False
        assert d["error"] == "failed"


class TestMediaStorage:
    def test_init_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            assert storage.media_path.exists()
            assert storage.images_path.exists()
            assert storage.videos_path.exists()
            assert storage.temp_path.exists()

    def test_validate_path_within_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, errors = storage.validate_path("media/images/test.png")
            assert ok is True

    def test_validate_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, errors = storage.validate_path("../../../etc/passwd")
            assert ok is False
            assert any("traversal" in e.lower() or "outside" in e.lower() for e in errors)

    def test_validate_path_absolute_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, errors = storage.validate_path("/etc/passwd")
            assert ok is False

    def test_validate_empty_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, errors = storage.validate_path("")
            assert ok is False

    def test_validate_file_type_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, errors = storage.validate_file_type("test.png", MediaType.IMAGE)
            assert ok is True

    def test_validate_file_type_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, errors = storage.validate_file_type("test.txt", MediaType.IMAGE)
            assert ok is False

    def test_validate_file_size_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, errors = storage.validate_file_size(1024, MediaType.IMAGE)
            assert ok is True

    def test_validate_file_size_too_large(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, errors = storage.validate_file_size(100 * 1024 * 1024, MediaType.IMAGE)
            assert ok is False

    def test_sanitize_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            result = storage.sanitize_filename("../../../bad.png")
            assert ".." not in result
            assert "/" not in result
            assert "\\" not in result

    def test_sanitize_empty_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            result = storage.sanitize_filename("")
            assert result.startswith("media_")

    def test_store_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, path, errors = storage.store_bytes(
                b"test data", MediaType.IMAGE, "test.png"
            )
            assert ok is True
            assert os.path.exists(path)

    def test_store_bytes_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, path, errors = storage.store_bytes(
                b"test video data", MediaType.VIDEO, "test.mp4"
            )
            assert ok is True
            assert "videos" in path

    def test_get_storage_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            path = storage.get_storage_path(MediaType.IMAGE, "test.png")
            assert "images" in str(path)

    def test_list_media_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            files = storage.list_media()
            assert files == []

    def test_list_media_with_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            storage.store_bytes(b"test", MediaType.IMAGE, "test.png")
            files = storage.list_media(MediaType.IMAGE)
            assert len(files) == 1

    def test_cleanup_temp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            storage.store_bytes(b"temp", MediaType.IMAGE, "temp.png")
            removed = storage.cleanup_temp(max_age_seconds=0)
            assert removed >= 0

    def test_get_storage_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            stats = storage.get_storage_stats()
            assert "image_count" in stats
            assert "video_count" in stats

    def test_delete_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, path, _ = storage.store_bytes(b"test", MediaType.IMAGE, "del.png")
            assert ok is True
            ok, errors = storage.delete_file(path)
            assert ok is True
            assert not os.path.exists(path)

    def test_get_file_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            ok, path, _ = storage.store_bytes(b"test", MediaType.IMAGE, "info.png")
            info = storage.get_file_info(path)
            assert info is not None
            assert info["name"] == "info.png"


class TestVisionProvider:
    def test_name_and_type(self):
        vp = VisionProvider()
        assert vp.name == "vision"
        assert vp.media_type == MediaType.IMAGE
        assert vp.is_available is True

    def test_validate_no_path(self):
        vp = VisionProvider()
        req = MediaRequest(media_type=MediaType.IMAGE)
        ok, errors = vp.validate_request(req)
        assert ok is False

    def test_validate_nonexistent_path(self):
        vp = VisionProvider()
        req = MediaRequest(media_type=MediaType.IMAGE, input_path="/nonexistent.png")
        ok, errors = vp.validate_request(req)
        assert ok is False

    def test_process_no_path(self):
        vp = VisionProvider()
        req = MediaRequest(media_type=MediaType.IMAGE)
        result = vp.process(req)
        assert result.success is False

    def test_get_info(self):
        vp = VisionProvider()
        info = vp.get_info()
        assert info["name"] == "vision"
        assert info["media_type"] == "image"


class TestStubVisionProvider:
    def test_init(self):
        svp = StubLocalVisionProvider()
        assert svp.name == "stub_local_vision"
        assert svp.is_available is False

    def test_init_with_model(self):
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            svp = StubLocalVisionProvider(model_path=f.name)
            assert svp.is_available is True


class TestImageGenProvider:
    def test_name_and_type(self):
        igp = ImageGenProvider()
        assert igp.name == "image_generate"
        assert igp.media_type == MediaType.IMAGE

    def test_validate_no_prompt(self):
        igp = ImageGenProvider()
        req = MediaRequest(media_type=MediaType.IMAGE)
        ok, errors = igp.validate_request(req)
        assert ok is False

    def test_validate_invalid_width(self):
        igp = ImageGenProvider()
        req = MediaRequest(media_type=MediaType.IMAGE, prompt="test", width=-1)
        ok, errors = igp.validate_request(req)
        assert ok is False

    def test_validate_invalid_height(self):
        igp = ImageGenProvider()
        req = MediaRequest(media_type=MediaType.IMAGE, prompt="test", height=99999)
        ok, errors = igp.validate_request(req)
        assert ok is False

    def test_validate_invalid_seed(self):
        igp = ImageGenProvider()
        req = MediaRequest(media_type=MediaType.IMAGE, prompt="test", seed=-1)
        ok, errors = igp.validate_request(req)
        assert ok is False

    def test_process_generates_file(self):
        igp = ImageGenProvider()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test.png")
            req = MediaRequest(
                media_type=MediaType.IMAGE,
                prompt="a test image",
                output_path=output_path,
                width=64, height=64,
            )
            result = igp.process(req)
            assert result.success is True
            assert result.output is not None
            assert os.path.exists(output_path)

    def test_process_invalid_request(self):
        igp = ImageGenProvider()
        req = MediaRequest(media_type=MediaType.IMAGE)
        result = igp.process(req)
        assert result.success is False


class TestVideoGenProvider:
    def test_name_and_type(self):
        vgp = VideoGenProvider()
        assert vgp.name == "video_generate"
        assert vgp.media_type == MediaType.VIDEO

    def test_validate_no_prompt(self):
        vgp = VideoGenProvider()
        req = MediaRequest(media_type=MediaType.VIDEO)
        ok, errors = vgp.validate_request(req)
        assert ok is False

    def test_validate_invalid_duration(self):
        vgp = VideoGenProvider()
        req = MediaRequest(media_type=MediaType.VIDEO, prompt="test", duration=-1)
        ok, errors = vgp.validate_request(req)
        assert ok is False

    def test_validate_duration_too_long(self):
        vgp = VideoGenProvider()
        req = MediaRequest(media_type=MediaType.VIDEO, prompt="test", duration=999)
        ok, errors = vgp.validate_request(req)
        assert ok is False

    def test_validate_invalid_frames(self):
        vgp = VideoGenProvider()
        req = MediaRequest(media_type=MediaType.VIDEO, prompt="test", num_frames=0)
        ok, errors = vgp.validate_request(req)
        assert ok is False

    def test_process_generates_file(self):
        vgp = VideoGenProvider()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test.mp4")
            req = MediaRequest(
                media_type=MediaType.VIDEO,
                prompt="a test video",
                output_path=output_path,
                width=64, height=64,
                duration=1.0, num_frames=5,
            )
            result = vgp.process(req)
            assert result.success is True
            assert result.output is not None
            assert os.path.exists(output_path)


class TestMediaRouter:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            assert len(router.list_providers()) == 0

    def test_register_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            vp = VisionProvider()
            ok = router.register_provider(vp)
            assert ok is True
            assert "vision" in router.list_provider_names()

    def test_register_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            vp = VisionProvider()
            router.register_provider(vp)
            ok = router.register_provider(vp)
            assert ok is False

    def test_unregister_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            vp = VisionProvider()
            router.register_provider(vp)
            ok = router.unregister_provider("vision")
            assert ok is True
            assert "vision" not in router.list_provider_names()

    def test_get_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            vp = VisionProvider()
            router.register_provider(vp)
            p = router.get_provider("vision")
            assert p is vp

    def test_select_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            vp = VisionProvider()
            igp = ImageGenProvider()
            router.register_provider(vp)
            router.register_provider(igp)
            selected = router.select_provider(MediaType.IMAGE)
            assert selected is not None

    def test_select_preferred(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            vp = VisionProvider()
            igp = ImageGenProvider()
            router.register_provider(vp)
            router.register_provider(igp)
            selected = router.select_provider(MediaType.IMAGE, preferred="image_generate")
            assert selected is igp

    def test_route_no_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            req = MediaRequest(media_type=MediaType.IMAGE, prompt="test")
            result = router.route(req)
            assert result.success is False
            assert "No provider" in result.error

    def test_route_with_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            igp = ImageGenProvider()
            router.register_provider(igp)
            output_path = str(Path(tmpdir) / "media" / "images" / "test.png")
            req = MediaRequest(
                media_type=MediaType.IMAGE,
                prompt="test image",
                output_path=output_path,
                width=64, height=64,
            )
            result = router.route(req)
            assert result.success is True

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            vp = VisionProvider()
            router.register_provider(vp)
            stats = router.get_stats()
            assert stats["provider_count"] == 1
            assert "total_requests" in stats


class TestImageAnalyzeTool:
    def test_name_and_description(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            tool = ImageAnalyzeTool(router)
            assert tool.name == "image_analyze"
            assert len(tool.description) > 0

    def test_validate_no_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            tool = ImageAnalyzeTool(router)
            ok, errors = tool.validate({})
            assert ok is False

    def test_validate_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            tool = ImageAnalyzeTool(router)
            ok, errors = tool.validate({"image_path": "/nonexistent.png"})
            assert ok is False

    def test_execute_no_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            tool = ImageAnalyzeTool(router)
            result = tool.execute({"image_path": "/nonexistent.png"})
            assert result.success is False


class TestImageGenerateTool:
    def test_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            tool = ImageGenerateTool(router)
            assert tool.name == "image_generate"

    def test_validate_no_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            tool = ImageGenerateTool(router)
            ok, errors = tool.validate({})
            assert ok is False

    def test_validate_invalid_width(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            tool = ImageGenerateTool(router)
            ok, errors = tool.validate({"prompt": "test", "width": -1})
            assert ok is False

    def test_execute_generates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            igp = ImageGenProvider()
            router.register_provider(igp)
            tool = ImageGenerateTool(router)
            result = tool.execute({
                "prompt": "a test image",
                "width": 64,
                "height": 64,
            })
            assert result.success is True


class TestVideoGenerateTool:
    def test_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            tool = VideoGenerateTool(router)
            assert tool.name == "video_generate"

    def test_validate_no_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            tool = VideoGenerateTool(router)
            ok, errors = tool.validate({})
            assert ok is False

    def test_validate_invalid_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            tool = VideoGenerateTool(router)
            ok, errors = tool.validate({"prompt": "test", "duration": -1})
            assert ok is False

    def test_execute_generates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            vgp = VideoGenProvider()
            router.register_provider(vgp)
            tool = VideoGenerateTool(router)
            result = tool.execute({
                "prompt": "a test video",
                "width": 64,
                "height": 64,
                "duration": 1.0,
                "num_frames": 5,
            })
            assert result.success is True


class TestMediaInfoTool:
    def test_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            tool = MediaInfoTool(storage)
            assert tool.name == "media_info"

    def test_validate_invalid_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            tool = MediaInfoTool(storage)
            ok, errors = tool.validate({"action": "invalid"})
            assert ok is False

    def test_validate_info_requires_filepath(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            tool = MediaInfoTool(storage)
            ok, errors = tool.validate({"action": "info"})
            assert ok is False

    def test_execute_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            tool = MediaInfoTool(storage)
            result = tool.execute({"action": "stats"})
            assert result.success is True

    def test_execute_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            tool = MediaInfoTool(storage)
            result = tool.execute({"action": "list"})
            assert result.success is True

    def test_execute_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            storage.store_bytes(b"test", MediaType.IMAGE, "test.png")
            tool = MediaInfoTool(storage)
            img_path = str(storage.images_path / "test.png")
            result = tool.execute({"action": "info", "filepath": img_path})
            assert result.success is True


class TestMediaIntegration:
    def test_full_image_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            igp = ImageGenProvider()
            router.register_provider(igp)

            output_path = str(Path(tmpdir) / "media" / "images" / "flow_test.png")
            req = MediaRequest(
                media_type=MediaType.IMAGE,
                prompt="a beautiful landscape",
                output_path=output_path,
                width=128, height=128,
                seed=42,
            )
            result = router.route(req)
            assert result.success is True
            assert result.output is not None
            assert os.path.exists(output_path)

            stats = router.get_stats()
            assert stats["total_requests"] == 1

    def test_full_video_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            vgp = VideoGenProvider()
            router.register_provider(vgp)

            output_path = str(Path(tmpdir) / "media" / "videos" / "flow_test.mp4")
            req = MediaRequest(
                media_type=MediaType.VIDEO,
                prompt="a sunset over mountains",
                output_path=output_path,
                width=128, height=128,
                duration=2.0, num_frames=10,
            )
            result = router.route(req)
            assert result.success is True
            assert result.output is not None
            assert os.path.exists(output_path)

    def test_provider_registration_and_listing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)

            vp = VisionProvider()
            igp = ImageGenProvider()
            vgp = VideoGenProvider()

            router.register_provider(vp)
            router.register_provider(igp)
            router.register_provider(vgp)

            providers = router.list_providers()
            assert len(providers) == 3

            image_providers = router.get_providers_for_type(MediaType.IMAGE)
            assert len(image_providers) == 2

            video_providers = router.get_providers_for_type(MediaType.VIDEO)
            assert len(video_providers) == 1

    def test_path_security_across_components(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)

            ok, errors = storage.validate_path("../../../etc/passwd")
            assert ok is False

            ok, errors = storage.validate_path("/etc/passwd")
            assert ok is False

            safe = storage.sanitize_filename("../../../bad.png")
            assert ".." not in safe
            assert "/" not in safe

    def test_media_tool_registration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MediaStorage(workspace_dir=tmpdir)
            router = MediaRouter(storage=storage)
            igp = ImageGenProvider()
            router.register_provider(igp)

            from agent.tools import ToolRegistry
            registry = ToolRegistry()

            tools = [
                ImageAnalyzeTool(router),
                ImageGenerateTool(router),
                VideoGenerateTool(router),
                MediaInfoTool(storage),
            ]

            for tool in tools:
                ok = registry.register(tool)
                assert ok is True

            assert registry.count() == 4

            names = registry.list_names()
            assert "image_analyze" in names
            assert "image_generate" in names
            assert "video_generate" in names
            assert "media_info" in names
