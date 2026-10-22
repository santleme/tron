"""Dependency-free artifact helpers shared by training and evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any


def canonical_json(value: Any, digest_field: str | None = None) -> bytes:
    """Serialize UTF-8 JSON with sorted keys and ECMAScript/JCS number spelling."""
    def check(item: Any) -> None:
        if isinstance(item, str):
            if unicodedata.normalize("NFC", item) != item:
                raise ValueError("non-NFC JSON string")
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                check(key)
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("non-finite JSON number")
            if item.is_integer() and abs(item) > 9007199254740991:
                raise ValueError("integer-valued JSON number outside interoperable safe range")
        elif item is None or type(item) in (bool, int, float):
            if type(item) is int and abs(item) > 9007199254740991:
                raise ValueError("JSON integer outside interoperable safe range")
        else:
            raise TypeError(f"unsupported JSON value: {type(item).__name__}")

    check(value)
    document = dict(value)
    if digest_field:
        document.pop(digest_field, None)

    def string(text: str) -> str:
        return json.dumps(text, ensure_ascii=False, separators=(",", ":"))

    def number(value: float) -> str:
        # Python's repr is a shortest-roundtrip representation, as is
        # ECMAScript Number::toString; normalize only their grammar/threshold
        # differences (notably 1.0, -0.0, exponent zero-padding, and 1e-6/1e21).
        if value == 0:
            return "0"
        negative = value < 0
        text = repr(abs(value)).lower()
        if "e" in text:
            mantissa, exponent_text = text.split("e", 1)
            exponent = int(exponent_text)
        else:
            mantissa, exponent = text, 0
        if "." in mantissa:
            integer, fraction = mantissa.split(".", 1)
            digits = integer + fraction
            decimal_position = len(integer) + exponent
        else:
            digits = mantissa
            decimal_position = len(mantissa) + exponent
        leading = len(digits) - len(digits.lstrip("0"))
        digits = digits.lstrip("0") or "0"
        decimal_position -= leading
        digits = digits.rstrip("0") or "0"
        magnitude = abs(value)
        if 1e-6 <= magnitude < 1e21:
            if decimal_position <= 0:
                rendered = "0." + "0" * (-decimal_position) + digits
            elif decimal_position >= len(digits):
                rendered = digits + "0" * (decimal_position - len(digits))
            else:
                rendered = digits[:decimal_position] + "." + digits[decimal_position:]
        else:
            scientific_exponent = decimal_position - 1
            fraction = digits[1:]
            rendered = digits[0] + ("." + fraction if fraction else "")
            rendered += "e" + ("+" if scientific_exponent >= 0 else "") + str(scientific_exponent)
        return ("-" if negative else "") + rendered

    def encode(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if type(item) is int:
            return str(item)
        if type(item) is float:
            return number(item)
        if isinstance(item, str):
            return string(item)
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(encode(part) for part in item) + "]"
        if isinstance(item, dict):
            return "{" + ",".join(string(key) + ":" + encode(item[key])
                                    for key in sorted(item)) + "}"
        raise TypeError(f"unsupported JSON value: {type(item).__name__}")

    return encode(document).encode("utf-8")


def digest_json(value: dict[str, Any], digest_field: str) -> str:
    return hashlib.sha256(canonical_json(value, digest_field)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or "\\" in value or "\x00" in value or ".." in path.parts:
        raise ValueError(f"unsafe artifact path: {value!r}")
    return value


def write_json(path: str | Path, value: dict[str, Any], digest_field: str | None = None) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if digest_field:
        value[digest_field] = digest_json(value, digest_field)
    target.write_bytes(canonical_json(value))
    return file_sha256(target)
