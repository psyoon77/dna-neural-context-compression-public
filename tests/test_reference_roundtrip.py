"""Round-trip checks for the pinned Project Nayuki Python codecs."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "third_party" / "reference-arithmetic-coding" / "python"


class ReferenceCodecRoundTripTests(unittest.TestCase):
    def test_reference_codecs_restore_every_byte(self) -> None:
        payload = bytes(range(256)) + b"\nACGTACGT\n" * 32
        codec_pairs = (
            ("arithmetic-compress.py", "arithmetic-decompress.py"),
            ("adaptive-arithmetic-compress.py", "adaptive-arithmetic-decompress.py"),
            ("ppm-compress.py", "ppm-decompress.py"),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "source.bin"
            source.write_bytes(payload)

            for compressor, decompressor in codec_pairs:
                with self.subTest(codec=compressor):
                    encoded = temporary / f"{compressor}.bin"
                    restored = temporary / f"{decompressor}.restored"
                    subprocess.run(
                        [sys.executable, str(REFERENCE / compressor), str(source), str(encoded)],
                        check=True,
                        cwd=REFERENCE,
                    )
                    subprocess.run(
                        [sys.executable, str(REFERENCE / decompressor), str(encoded), str(restored)],
                        check=True,
                        cwd=REFERENCE,
                    )
                    self.assertEqual(payload, restored.read_bytes())


if __name__ == "__main__":
    unittest.main()
