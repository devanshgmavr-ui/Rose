"""Phase 11 - Multimodal Agent Integration Tests.

Tests that the vision system is properly integrated into the agent's
tool selection and autonomous loop.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.core.config import Config
from agent.media.base import MediaType, MediaRequest, MediaResult
from agent.media.vision import VisionProvider, VisionResult, DetectedElement, BoundingBox, VisionConfidence
from agent.media.real_vision import RealVisionProvider, ImagePreprocessor, ImageMetadata, ColorInfo, ImageAnalysis
from agent.media.analyzer import VisionAnalyzer
from agent.media.vision_tool import VisionAnalyzeTool
from agent.media.storage import MediaStorage
from agent.orchestration.tool_catalog import build_tool_catalog, get_tools_for_request
from agent.orchestration.tool_selector import IntentClassifier, ToolSelector
from agent.orchestration.capability_analyzer import CapabilityAnalyzer, CAPABILITY_DEFINITIONS
from agent.orchestration.tool_scorer import ToolScorer, CAPABILITY_TOOLS_MAP


class TestConfigVisionProviderType:
    def test_default_is_local(self):
        config = Config()
        assert config.vision_provider_type == "local"

    def test_accepts_real(self):
        config = Config()
        config.vision_provider_type = "real"
        assert config.vision_provider_type == "real"

    def test_vision_settings(self):
        config = Config()
        assert config.vision_max_image_size_mb == 20
        assert config.vision_max_image_width == 4096
        assert config.vision_max_image_height == 4096
        assert config.vision_max_elements == 100
        assert config.vision_analysis_timeout == 30000


class TestRealVisionProviderIntegration:
    def test_init_with_config_params(self):
        provider = RealVisionProvider(
            max_image_size_mb=10,
            max_image_width=2048,
            max_image_height=2048,
        )
        assert provider._max_image_size_mb == 10
        assert provider._max_image_width == 2048
        assert provider._max_image_height == 2048

    def test_capabilities_include_all_expected(self):
        provider = RealVisionProvider()
        caps = provider.get_capabilities()
        expected_keys = ["image_analysis", "color_analysis", "region_detection",
                         "ocr", "multimodal", "metadata_extraction", "preprocessing"]
        for key in expected_keys:
            assert key in caps, f"Missing capability: {key}"

    def test_analyze_produces_valid_vision_result(self):
        try:
            from PIL import Image

            provider = RealVisionProvider()
            provider.initialize()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (200, 150), color=(255, 128, 0))
                img.save(f)
                path = f.name

            try:
                request = MediaRequest(
                    media_type=MediaType.IMAGE,
                    input_path=path,
                    prompt="What is in this image?",
                )
                result = provider._analyze_image(request)
                assert result.success is True
                assert result.image_width == 200
                assert result.image_height == 150
                assert result.provider == "real_vision"
                # VisionResult should have description
                assert result.description != ""
            finally:
                os.unlink(path)

        except ImportError:
            pytest.skip("Pillow not installed")


class TestVisionAnalyzerIntegration:
    def test_analyzer_with_real_provider(self):
        try:
            from PIL import Image

            provider = RealVisionProvider()
            provider.initialize()

            analyzer = VisionAnalyzer(vision_provider=provider)
            assert analyzer.is_available
            assert analyzer.provider_name == "real_vision"

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (100, 100), color=(0, 200, 100))
                img.save(f)
                path = f.name

            try:
                result = analyzer.analyze(path)
                assert result.success is True
                assert result.provider == "real_vision"
            finally:
                os.unlink(path)

        except ImportError:
            pytest.skip("Pillow not installed")

    def test_analyzer_describe(self):
        try:
            from PIL import Image

            provider = RealVisionProvider()
            provider.initialize()

            analyzer = VisionAnalyzer(vision_provider=provider)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (80, 60), color=(50, 50, 200))
                img.save(f)
                path = f.name

            try:
                text = analyzer.describe_image(path)
                assert "[BEGIN UNTRUSTED VISUAL CONTENT]" in text
                assert "[END UNTRUSTED VISUAL CONTENT]" in text
            finally:
                os.unlink(path)

        except ImportError:
            pytest.skip("Pillow not installed")


class TestToolCatalogVisionEntries:
    def test_vision_analyze_in_catalog(self):
        catalog = build_tool_catalog()
        assert "vision_analyze" in catalog
        meta = catalog["vision_analyze"]
        assert meta.category == "vision"
        assert "analyze" in meta.actions
        assert "describe" in meta.actions

    def test_visual_ground_in_catalog(self):
        catalog = build_tool_catalog()
        assert "visual_ground" in catalog
        meta = catalog["visual_ground"]
        assert meta.category == "vision"
        assert "ground" in meta.actions

    def test_vision_keywords_match(self):
        catalog = build_tool_catalog()
        tools = get_tools_for_request("analyze this screenshot", catalog)
        names = [t.name for t in tools]
        assert "vision_analyze" in names

    def test_visual_ground_keywords_match(self):
        catalog = build_tool_catalog()
        tools = get_tools_for_request("find button", catalog)
        names = [t.name for t in tools]
        assert "visual_ground" in names

    def test_ocr_keywords_match(self):
        catalog = build_tool_catalog()
        tools = get_tools_for_request("extract text from image", catalog)
        names = [t.name for t in tools]
        assert "vision_analyze" in names

    def test_color_analysis_keywords_match(self):
        catalog = build_tool_catalog()
        tools = get_tools_for_request("detect colors in image", catalog)
        names = [t.name for t in tools]
        assert "vision_analyze" in names


class TestIntentClassifierVisionPatterns:
    def test_vision_analyze_patterns(self):
        classifier = IntentClassifier()
        matches = classifier.classify("analyze this screenshot")
        assert any(m.tool_name == "vision_analyze" for m in matches)

    def test_visual_ground_patterns(self):
        classifier = IntentClassifier()
        matches = classifier.classify("find the submit button")
        assert any(m.tool_name == "visual_ground" for m in matches)

    def test_describe_image_patterns(self):
        classifier = IntentClassifier()
        matches = classifier.classify("describe this image")
        assert any(m.tool_name == "vision_analyze" for m in matches)

    def test_look_at_patterns(self):
        classifier = IntentClassifier()
        matches = classifier.classify("look at this picture")
        assert any(m.tool_name == "vision_analyze" for m in matches)


class TestCapabilityAnalyzerVisionCapabilities:
    def test_vision_analysis_capability(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("analyze this image")
        cap_names = result.get_capability_names()
        assert "vision_analysis" in cap_names

    def test_visual_grounding_capability(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("find button")
        cap_names = result.get_capability_names()
        assert "visual_grounding" in cap_names

    def test_text_transcription_capability(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("transcribe text from image")
        cap_names = result.get_capability_names()
        assert "text_transcription" in cap_names

    def test_verification_with_vision(self):
        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze("verify the task by checking the screen")
        cap_names = result.get_capability_names()
        assert "verification" in cap_names

    def test_vision_definition_exists(self):
        assert "vision_analysis" in CAPABILITY_DEFINITIONS
        assert "visual_grounding" in CAPABILITY_DEFINITIONS
        assert "text_transcription" in CAPABILITY_DEFINITIONS


class TestToolScorerVisionMapping:
    def test_vision_analysis_maps_to_tools(self):
        assert "vision_analysis" in CAPABILITY_TOOLS_MAP
        tools = CAPABILITY_TOOLS_MAP["vision_analysis"]
        assert "vision_analyze" in tools

    def test_visual_grounding_maps_to_tools(self):
        assert "visual_grounding" in CAPABILITY_TOOLS_MAP
        tools = CAPABILITY_TOOLS_MAP["visual_grounding"]
        assert "visual_ground" in tools

    def test_text_transcription_maps_to_tools(self):
        assert "text_transcription" in CAPABILITY_TOOLS_MAP
        tools = CAPABILITY_TOOLS_MAP["text_transcription"]
        assert "vision_analyze" in tools
        assert "keyboard" in tools

    def test_verification_maps_to_vision(self):
        assert "verification" in CAPABILITY_TOOLS_MAP
        tools = CAPABILITY_TOOLS_MAP["verification"]
        assert "vision_analyze" in tools
        assert "screen_capture" in tools

    def test_scorer_selects_vision_for_image_analysis(self):
        from agent.orchestration.capability_analyzer import Capability

        scorer = ToolScorer()
        capabilities = [Capability(name="vision_analysis", description="Analyze images", confidence=0.9)]
        result = scorer.select_tool(capabilities)
        assert result.selected_tool is not None
        assert result.selected_tool.tool_name in ("vision_analyze", "image_analyze")


class TestVisionToolWithRealProvider:
    def test_vision_tool_execute_with_real_provider(self):
        try:
            from PIL import Image

            provider = RealVisionProvider()
            provider.initialize()

            analyzer = VisionAnalyzer(vision_provider=provider)
            storage = MediaStorage(workspace_dir="workspace")

            tool = VisionAnalyzeTool(
                vision_analyzer=analyzer,
                media_storage=storage,
                vision_enabled=True,
                workspace_dir="workspace",
            )

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (120, 90), color=(100, 200, 50))
                img.save(f)
                path = f.name

            try:
                result = tool.execute({"image_path": path, "action": "describe"})
                assert result.success is True
                assert result.output is not None
                assert "UNTRUSTED" in result.output
            finally:
                os.unlink(path)

        except ImportError:
            pytest.skip("Pillow not installed")

    def test_vision_tool_analyze_action(self):
        try:
            from PIL import Image

            provider = RealVisionProvider()
            provider.initialize()

            analyzer = VisionAnalyzer(vision_provider=provider)
            storage = MediaStorage(workspace_dir="workspace")

            # Create workspace dir and put test image inside it
            ws = Path("workspace")
            ws.mkdir(exist_ok=True)

            tool = VisionAnalyzeTool(
                vision_analyzer=analyzer,
                media_storage=storage,
                vision_enabled=True,
                workspace_dir=str(ws),
            )

            img_path = ws / "test_analyze.png"
            try:
                img = Image.new("RGB", (150, 100), color=(200, 100, 150))
                img.save(str(img_path))

                result = tool.execute({
                    "image_path": str(img_path),
                    "action": "analyze",
                    "prompt": "What colors are present?",
                })
                assert result.success is True
            finally:
                if img_path.exists():
                    img_path.unlink()

        except ImportError:
            pytest.skip("Pillow not installed")


class TestEndToEndVisionFlow:
    def test_screenshot_analyze_flow(self):
        try:
            from PIL import Image

            provider = RealVisionProvider()
            provider.initialize()

            analyzer = VisionAnalyzer(vision_provider=provider)

            # Simulate a screenshot
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (1920, 1080), color=(30, 30, 30))
                # Add a red button-like region
                for x in range(100, 200):
                    for y in range(100, 140):
                        img.putpixel((x, y), (200, 50, 50))
                img.save(f)
                path = f.name

            try:
                # Step 1: Analyze
                result = analyzer.analyze(path, prompt="Find any buttons")
                assert result.success is True
                assert result.image_width == 1920
                assert result.image_height == 1080

                # Step 2: Get text description
                text = analyzer.describe_image(path)
                assert "UNTRUSTED" in text

            finally:
                os.unlink(path)

        except ImportError:
            pytest.skip("Pillow not installed")

    def test_multiple_vision_requests(self):
        try:
            from PIL import Image

            provider = RealVisionProvider()
            provider.initialize()

            analyzer = VisionAnalyzer(vision_provider=provider)

            paths = []
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    img = Image.new("RGB", (100, 100), color=(i * 80, 100, 200 - i * 50))
                    img.save(f)
                    paths.append(f.name)

            try:
                for path in paths:
                    result = analyzer.analyze(path)
                    assert result.success is True

                stats = analyzer.stats
                assert stats["request_count"] == 3
                assert stats["total_time"] > 0
            finally:
                for p in paths:
                    os.unlink(p)

        except ImportError:
            pytest.skip("Pillow not installed")
