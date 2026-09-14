"""Streaming helpers for NeuroOracle's multi-gigabyte graph JSON."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TextIO


CONCEPTS_MARKER = '"concepts": {'
EDGES_MARKER = '"edges": ['
CONCEPTS_MARKERS = (CONCEPTS_MARKER, '"concepts":{')
EDGES_MARKERS = (EDGES_MARKER, '"edges":[')


class IncrementalJsonReader:
    def __init__(
        self,
        handle: TextIO,
        initial: str = "",
        chunk_size: int = 1024 * 1024,
    ) -> None:
        self.handle = handle
        self.buffer = initial
        self.position = 0
        self.chunk_size = chunk_size
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _compact(self) -> None:
        if self.position > self.chunk_size:
            self.buffer = self.buffer[self.position :]
            self.position = 0

    def _fill(self) -> bool:
        self._compact()
        chunk = self.handle.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def skip_whitespace(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or not self._fill():
                return

    def peek(self) -> str:
        self.skip_whitespace()
        if self.position >= len(self.buffer):
            raise EOFError("unexpected end of graph JSON")
        return self.buffer[self.position]

    def expect(self, character: str) -> None:
        if self.peek() != character:
            raise ValueError(
                f"expected {character!r}, found "
                f"{self.buffer[self.position:self.position + 20]!r}"
            )
        self.position += 1

    def value(self) -> Any:
        self.skip_whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof or not self._fill():
                    raise
                continue
            self.position = end
            return value

    def unread_text(self) -> str:
        return self.buffer[self.position :]


def _first_marker(text: str, markers: tuple[str, ...]) -> tuple[int, str] | None:
    matches = (
        (index, marker)
        for marker in markers
        if (index := text.find(marker)) >= 0
    )
    return min(matches, default=None, key=lambda value: value[0])


def _reader_at_concepts(handle: TextIO) -> tuple[str, IncrementalJsonReader]:
    prefix = ""
    match = _first_marker(prefix, CONCEPTS_MARKERS)
    while match is None:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            raise ValueError("concepts object not found in graph")
        prefix += chunk
        if len(prefix) > 64 * 1024 * 1024:
            raise ValueError("graph metadata prefix is unexpectedly large")
        match = _first_marker(prefix, CONCEPTS_MARKERS)
    marker_index, marker = match
    marker_end = marker_index + len(marker)
    return prefix[:marker_end], IncrementalJsonReader(handle, prefix[marker_end:])


def _reader_at_array(
    handle: TextIO,
    markers: str | tuple[str, ...],
    *,
    output: TextIO | None = None,
) -> IncrementalJsonReader:
    """Seek to a top-level array marker without retaining its prefix in memory.

    ``edges`` follows the multi-gigabyte ``concepts`` object in the formal KG,
    so the prefix must be streamed to the destination (or discarded) rather
    than accumulated as one Python string.
    """

    if isinstance(markers, str):
        markers = (markers,)
    carry = ""
    overlap = max(0, max(map(len, markers)) - 1)
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            raise ValueError(f"{markers!r} array not found in graph")
        combined = carry + chunk
        match = _first_marker(combined, markers)
        if match is not None:
            marker_index, marker = match
            if output is not None:
                output.write(combined[: marker_index + len(marker)])
            initial = combined[marker_index + len(marker) :]
            return IncrementalJsonReader(handle, initial)
        safe_length = max(0, len(combined) - overlap)
        if output is not None and safe_length:
            output.write(combined[:safe_length])
        carry = combined[safe_length:]


def iter_concepts(graph_path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with graph_path.open("r", encoding="utf-8") as handle:
        _, reader = _reader_at_concepts(handle)
        while True:
            if reader.peek() == "}":
                reader.expect("}")
                return
            node_id = reader.value()
            if not isinstance(node_id, str):
                raise ValueError("concept key is not a string")
            reader.expect(":")
            node = reader.value()
            if not isinstance(node, dict):
                raise ValueError(f"concept {node_id!r} is not an object")
            yield node_id, node
            delimiter = reader.peek()
            if delimiter == ",":
                reader.expect(",")
            elif delimiter == "}":
                reader.expect("}")
                return
            else:
                raise ValueError(f"unexpected concepts delimiter: {delimiter!r}")


def iter_edges(graph_path: Path) -> Iterator[dict[str, Any]]:
    """Yield edge objects without loading concepts or the full graph."""

    with graph_path.open("r", encoding="utf-8") as handle:
        reader = _reader_at_array(handle, EDGES_MARKERS)
        while True:
            if reader.peek() == "]":
                reader.expect("]")
                return
            edge = reader.value()
            if not isinstance(edge, dict):
                raise ValueError("graph edge is not an object")
            yield edge
            delimiter = reader.peek()
            if delimiter == ",":
                reader.expect(",")
            elif delimiter == "]":
                reader.expect("]")
                return
            else:
                raise ValueError(f"unexpected edges delimiter: {delimiter!r}")


def rewrite_concepts(
    graph_path: Path,
    output_path: Path,
    transform: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> int:
    """Rewrite concept objects while byte-streaming the rest of the graph.

    The destination must differ from the source.  The caller is responsible for
    validating the completed JSON and atomically replacing the source.
    """

    if graph_path.resolve() == output_path.resolve():
        raise ValueError("streaming graph rewrite requires a distinct output path")
    transformed = 0
    with graph_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        prefix, reader = _reader_at_concepts(source)
        output.write(prefix)
        first = True
        while True:
            if reader.peek() == "}":
                reader.expect("}")
                break
            node_id = reader.value()
            if not isinstance(node_id, str):
                raise ValueError("concept key is not a string")
            reader.expect(":")
            node = reader.value()
            if not isinstance(node, dict):
                raise ValueError(f"concept {node_id!r} is not an object")
            rewritten = transform(node_id, node)
            if not isinstance(rewritten, dict):
                raise TypeError(f"transform returned non-object for {node_id!r}")
            if not first:
                output.write(",")
            first = False
            output.write(json.dumps(node_id, ensure_ascii=False))
            output.write(":")
            output.write(json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")))
            transformed += 1

            delimiter = reader.peek()
            if delimiter == ",":
                reader.expect(",")
            elif delimiter == "}":
                reader.expect("}")
                break
            else:
                raise ValueError(f"unexpected concepts delimiter: {delimiter!r}")

        output.write("}")
        output.write(reader.unread_text())
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    return transformed


def rewrite_edges(
    graph_path: Path,
    output_path: Path,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> int:
    """Rewrite edge objects while streaming the preceding concepts object."""

    if graph_path.resolve() == output_path.resolve():
        raise ValueError("streaming graph rewrite requires a distinct output path")
    transformed = 0
    with graph_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        reader = _reader_at_array(source, EDGES_MARKERS, output=output)
        first = True
        while True:
            if reader.peek() == "]":
                reader.expect("]")
                break
            edge = reader.value()
            if not isinstance(edge, dict):
                raise ValueError("graph edge is not an object")
            rewritten = transform(edge)
            if not isinstance(rewritten, dict):
                raise TypeError("edge transform returned a non-object")
            if not first:
                output.write(",")
            first = False
            output.write(json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")))
            transformed += 1

            delimiter = reader.peek()
            if delimiter == ",":
                reader.expect(",")
            elif delimiter == "]":
                reader.expect("]")
                break
            else:
                raise ValueError(f"unexpected edges delimiter: {delimiter!r}")

        output.write("]")
        output.write(reader.unread_text())
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    return transformed


def is_claim_node(node_id: str, node: dict[str, Any]) -> bool:
    metadata = node.get("metadata") or {}
    return (
        node_id.startswith("CLM:")
        or "claim" in (node.get("domain_tags") or [])
        or bool(metadata.get("source_paper") and metadata.get("subject_name"))
    )


__all__ = [
    "CONCEPTS_MARKER",
    "CONCEPTS_MARKERS",
    "EDGES_MARKER",
    "EDGES_MARKERS",
    "IncrementalJsonReader",
    "is_claim_node",
    "iter_concepts",
    "iter_edges",
    "rewrite_concepts",
    "rewrite_edges",
]
