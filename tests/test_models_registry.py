from __future__ import annotations

import pytest

from f1_rag.errors import RegistryError
from f1_rag.models import RegulationSection, make_chunk_id
from f1_rag.registry import Registry


def test_chunk_id_is_deterministic():
    a = make_chunk_id("section-a.pdf", "A1.1.1", 0)
    b = make_chunk_id("section-a.pdf", "A1.1.1", 0)
    assert a == b
    assert a != make_chunk_id("section-a.pdf", "A1.1.1", 1)
    assert a != make_chunk_id("section-b.pdf", "A1.1.1", 0)


def test_citation_label(sample_meta):
    from tests.conftest import make_chunk

    chunk = make_chunk("cid", "some text", article_number="D3.1", section=RegulationSection.D, page=12)
    label = chunk.citation_label()
    assert "D3.1" in label
    assert "Section D" in label


def test_registry_register_and_create():
    reg: Registry = Registry("thing")

    @reg.register("foo")
    class Foo:
        def __init__(self, x=1):
            self.x = x

    assert "foo" in reg
    assert reg.available() == ["foo"]
    assert reg.create("foo", x=5).x == 5


def test_registry_duplicate_and_missing():
    reg: Registry = Registry("thing")

    @reg.register("foo")
    def _factory():
        return 1

    with pytest.raises(RegistryError):
        reg.register("foo")(lambda: 2)
    with pytest.raises(RegistryError):
        reg.create("missing")
