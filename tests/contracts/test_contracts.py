"""Focused schema, canonical digest, and seed-vector contract checks."""
import hashlib
import json
import struct
import unittest
import unicodedata
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError


ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "schemas"
FIXTURES = ROOT / "shared" / "fixtures"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(schema_name, document):
    schema = load_json(SCHEMAS / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)


def canonical_value(value):
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise ValueError("non-NFC JSON string")
        return value
    if isinstance(value, list):
        return [canonical_value(v) for v in value]
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if unicodedata.normalize("NFC", key) != key:
                raise ValueError("non-NFC JSON object key")
            normalized[key] = canonical_value(item)
        return normalized
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        raise ValueError("non-finite JSON number")
    return value


def canonical_json(value, excluded_digest_field):
    value = canonical_value(value)
    value.pop(excluded_digest_field, None)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


MASK64 = (1 << 64) - 1
PCG_MULTIPLIER = 6364136223846793005


def derive_pcg(master_seed, label, indices):
    preimage = (f"lc-arena/seed/v1/{label}".encode("utf-8") + b"\0"
                + struct.pack("<Q", int(master_seed))
                + b"".join(struct.pack("<Q", index) for index in indices))
    digest = hashlib.sha256(preimage).digest()
    initstate, initseq = struct.unpack("<QQ", digest[:16])
    increment = ((initseq << 1) | 1) & MASK64
    state = 0

    def next_u32():
        nonlocal state
        old = state
        state = (old * PCG_MULTIPLIER + increment) & MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rotation = old >> 59
        return ((xorshifted >> rotation) | (xorshifted << ((-rotation) & 31))) & 0xFFFFFFFF

    next_u32()
    state = (state + initstate) & MASK64
    next_u32()
    return preimage, digest.hex(), initstate, initseq, [next_u32() for _ in range(5)]


class SchemaContracts(unittest.TestCase):
    def test_valid_golden_artifacts(self):
        validate("lc-run-1.schema.json", load_json(FIXTURES / "lc-run-1.valid.json"))
        validate("lc-replay-1.schema.json", load_json(FIXTURES / "lc-replay-1.valid.json"))
        validate("lc-eval-1.schema.json", load_json(FIXTURES / "lc-eval-1.valid.json"))
        validate("checkpoint-metadata-1.schema.json", load_json(FIXTURES / "checkpoint-metadata-1.valid.json"))
        validate("collision-transitions-1.schema.json", load_json(FIXTURES / "collision-transitions-1.json"))

    def test_evaluation_has_exact_seed_seat_sample_matrix_and_summary_counts(self):
        report = load_json(FIXTURES / "lc-eval-1.valid.json")
        expected = {(str(seed), seat) for seed in range(32) for seat in (0, 1)}
        self.assertEqual({m["opponent_id"] for m in report["matchups"]}, {"open-space", "step-0"})
        for matchup in report["matchups"]:
            self.assertEqual({(game["seed"], game["target_seat"]) for game in matchup["games"]}, expected)
            summary = matchup["summary"]
            self.assertEqual(summary["wins"] + summary["draws"] + summary["losses"], 64)

    def test_run_pins_paired_checkpoint_formats_and_replay_tick_integrity(self):
        run = load_json(FIXTURES / "lc-run-1.valid.json")
        self.assertEqual({item["format"] for item in run["checkpoints"]}, {"pytorch", "onnx"})
        self.assertEqual(len({item["checkpoint_id"] for item in run["checkpoints"]}), 1)
        self.assertNotIn("manifest.json", {item["path"] for item in run["files"]})
        replay = load_json(FIXTURES / "lc-replay-1.valid.json")
        self.assertEqual(replay["tick_count"], len(replay["ticks"]))
        self.assertEqual([tick["tick"] for tick in replay["ticks"]], list(range(replay["tick_count"])))
        self.assertEqual(replay["ticks"][-1]["terminal_causes"], ["agent_0_dead", "agent_1_dead"])

    def test_digest_fields_are_canonical_and_excluded_from_their_own_hash(self):
        cases = [
            ("lc-run-1.valid.json", "manifest_sha256"),
            ("lc-replay-1.valid.json", "replay_sha256"),
            ("lc-eval-1.valid.json", "report_sha256"),
            ("checkpoint-metadata-1.valid.json", "metadata_sha256"),
        ]
        for filename, digest_field in cases:
            with self.subTest(filename=filename):
                document = load_json(FIXTURES / filename)
                canonical = canonical_json(document, digest_field).encode("utf-8")
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), document[digest_field])
                document["unrelated_mutation"] = True
                self.assertNotEqual(hashlib.sha256(canonical_json(document, digest_field).encode()).hexdigest(), document[digest_field])

    def test_rejects_unknown_version_missing_fields_and_unsafe_paths(self):
        doc = load_json(FIXTURES / "lc-run-1.valid.json")
        doc["format"] = "lc-run/99"
        with self.assertRaises(ValidationError):
            validate("lc-run-1.schema.json", doc)
        doc = load_json(FIXTURES / "lc-run-1.valid.json")
        del doc["master_seed"]
        with self.assertRaises(ValidationError):
            validate("lc-run-1.schema.json", doc)
        doc = load_json(FIXTURES / "lc-run-1.valid.json")
        doc["checkpoints"][0]["file"] = "../secret.pt"
        with self.assertRaises(ValidationError):
            validate("lc-run-1.schema.json", doc)

    def test_rejects_unsafe_numeric_seed_and_digest_spelling(self):
        doc = load_json(FIXTURES / "lc-run-1.valid.json")
        doc["master_seed"] = 9007199254740992
        with self.assertRaises(ValidationError):
            validate("lc-run-1.schema.json", doc)
        doc = load_json(FIXTURES / "lc-replay-1.valid.json")
        doc["replay_sha256"] = "A" * 64
        with self.assertRaises(ValidationError):
            validate("lc-replay-1.schema.json", doc)


class CanonicalDigest(unittest.TestCase):
    def test_canonical_utf8_bytes_and_digest_are_pinned(self):
        fixture = load_json(FIXTURES / "canonical-digest-1.json")
        for vector in fixture["vectors"]:
            canonical = canonical_json(vector["document"], vector["digest_field"])
            self.assertEqual(canonical.encode("utf-8").decode("utf-8"), vector["canonical_utf8"])
            self.assertEqual(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), vector["sha256"])

    def test_digest_field_is_excluded_and_nonfinite_fails(self):
        obj = {"digest": "not-hashed", "value": 0.5}
        self.assertEqual(canonical_json(obj, "digest"), '{"value":0.5}')
        with self.assertRaises(ValueError):
            canonical_json({"value": float("nan")}, "digest")
        with self.assertRaises(ValueError):
            canonical_json({"text": "Cafe\u0301"}, "digest")


class SeedVectors(unittest.TestCase):
    def test_sha256_domain_separated_pcg_vectors(self):
        vectors = load_json(FIXTURES / "seed-vectors-1.json")["vectors"]
        for vector in vectors:
            preimage, digest, initstate, initseq, outputs = derive_pcg(vector["master_seed"], vector["label"], vector["indices"])
            self.assertEqual(preimage.hex(), vector["preimage_hex"])
            self.assertEqual(digest, vector["sha256"])
            self.assertEqual(str(initstate), vector["initstate"])
            self.assertEqual(str(initseq), vector["initseq"])
            self.assertEqual(outputs, vector["pcg32"])

    def test_distinct_stream_labels_are_independent(self):
        v = load_json(FIXTURES / "seed-vectors-1.json")["vectors"]
        self.assertNotEqual(v[1]["sha256"], v[2]["sha256"])
        self.assertNotEqual(v[0]["pcg32"], v[3]["pcg32"])

    def test_u64_wire_boundary(self):
        doc = load_json(FIXTURES / "lc-replay-1.valid.json")
        doc["master_seed"] = "18446744073709551615"
        validate("lc-replay-1.schema.json", doc)
        doc["master_seed"] = "18446744073709551616"
        with self.assertRaises(ValidationError):
            validate("lc-replay-1.schema.json", doc)


if __name__ == "__main__":
    unittest.main()
