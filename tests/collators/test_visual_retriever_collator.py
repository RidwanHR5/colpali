from typing import Generator, cast

import pytest
from PIL import Image

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.models import ColModernVBertProcessor


class TestColModernVBertCollator:
    @pytest.fixture(scope="class")
    def colmodernvbert_processor_path(self) -> str:
        return "ModernVBERT/colmodernvbert"

    @pytest.fixture(scope="class")
    def processor_from_pretrained(
        self, colmodernvbert_processor_path: str
    ) -> Generator[ColModernVBertProcessor, None, None]:
        yield cast(ColModernVBertProcessor, ColModernVBertProcessor.from_pretrained(colmodernvbert_processor_path))

    @pytest.fixture(scope="class")
    def colmodernvbert_collator(
        self, processor_from_pretrained: ColModernVBertProcessor
    ) -> Generator[VisualRetrieverCollator, None, None]:
        yield VisualRetrieverCollator(processor=processor_from_pretrained)

    def test_colmodernvbert_collator_call(self, colmodernvbert_collator: VisualRetrieverCollator):
        example_image = Image.new("RGB", (16, 16), color="red")
        examples = [
            {"query": "What is this?", "pos_target": example_image},
        ]

        result = colmodernvbert_collator(examples)

        assert isinstance(result, dict)
        assert "doc_input_ids" in result
        assert "doc_attention_mask" in result
        assert "doc_pixel_values" in result
        assert "query_input_ids" in result
        assert "query_attention_mask" in result

    def test_colmodernvbert_collator_call_with_neg_images(self, colmodernvbert_collator: VisualRetrieverCollator):
        example_image = Image.new("RGB", (16, 16), color="red")
        neg_image = Image.new("RGB", (16, 16), color="blue")
        examples = [
            {
                "query": "What is this?",
                "pos_target": example_image,
                "neg_target": neg_image,
            },
        ]

        result = colmodernvbert_collator(examples)

        assert isinstance(result, dict)
        assert "doc_input_ids" in result
        assert "doc_attention_mask" in result
        assert "doc_pixel_values" in result
        assert "query_input_ids" in result
        assert "query_attention_mask" in result
        assert "neg_doc_input_ids" in result
        assert "neg_doc_attention_mask" in result
        assert "neg_doc_pixel_values" in result
