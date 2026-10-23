"""Focused tests for the fail-closed candidate audit helpers."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import audit_candidate_donor as audit_module
from audit_candidate_donor import group_id, infer_mapping, locate_label, parse_mapping


class CandidateAuditTests(unittest.TestCase):
    def test_known_semantics_are_inferred_but_generic_vegetation_is_not(self):
        mapping = infer_mapping(["water hyacinth", "aquatic vegetation", "ent_litter"], {})
        self.assertEqual(mapping["water hyacinth"], "hyacinth")
        self.assertEqual(mapping["ent_litter"], "entangled_plastic")
        self.assertNotIn("aquatic vegetation", mapping)

    def test_explicit_mapping_is_normalized(self):
        self.assertEqual(parse_mapping(["River Vegetation=hyacinth"]), {"river vegetation": "hyacinth"})

    def test_roboflow_suffix_and_video_frames_share_groups(self):
        root = Path("C:/candidate")
        first = Path("C:/candidate/train/images/IMG_42_frame_00001_jpg.rf.abc1234567890abc1234567890abc123.jpg")
        second = Path("C:/candidate/test/images/IMG_42_frame_00002_jpg.rf.def4567890abc1234567890abc123456.jpg")
        self.assertEqual(group_id(first, root, "auto"), group_id(second, root, "auto"))

    def test_images_labels_layout_is_resolved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "train" / "images" / "one.jpg"
            label = root / "train" / "labels" / "one.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            found, ambiguous = locate_label(image, root)
            self.assertEqual(found, label.resolve())
            self.assertFalse(ambiguous)

    def test_new_hyacinth_fixture_can_pass_without_merging(self):
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover - repository environment has Pillow
            self.skipTest("Pillow is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            repo = temp_root / "repo"
            candidate = temp_root / "candidate"
            output = temp_root / "audit"
            (repo / "data" / "processed").mkdir(parents=True)
            (candidate / "images").mkdir(parents=True)
            (candidate / "labels").mkdir(parents=True)
            (candidate / "classes.txt").write_text("water_hyacinth\n", encoding="utf-8")
            incumbent = repo / "data" / "processed" / "incumbent.jpg"
            new_image = candidate / "images" / "new.jpg"
            Image.new("RGB", (64, 64), (10, 180, 40)).save(incumbent)
            patterned = Image.new("RGB", (64, 64), "white")
            draw = ImageDraw.Draw(patterned)
            for y in range(0, 64, 8):
                for x in range(0, 64, 8):
                    if (x // 8 + y // 8) % 2:
                        draw.rectangle((x, y, x + 7, y + 7), fill="navy")
            patterned.save(new_image)
            (candidate / "labels" / "new.txt").write_text(
                "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
            args = Namespace(
                candidate_root=str(candidate),
                slug="fixture",
                source_url="https://example.invalid/fixture",
                license="CC BY 4.0",
                version="v1",
                map_class=[],
                group_mode="per-image",
                allow_ambiguous_grouping=False,
                phash_threshold=8,
                allow_empty_labels=False,
                output=str(output),
            )
            with patch.object(audit_module, "ROOT", repo):
                result = audit_module.audit(args)
            self.assertEqual(result, 0)
            report = (output / "candidate_audit.json").read_text(encoding="utf-8")
            self.assertIn('"status": "READY_FOR_MERGE"', report)


if __name__ == "__main__":
    unittest.main()
