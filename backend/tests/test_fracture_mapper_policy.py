import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fracture_mapper import map_fractures_to_3d  # noqa: E402


class FractureMapperPolicyTests(unittest.TestCase):
    def _job(self, root):
        job = Path(root) / "job"
        (job / "sparse" / "0").mkdir(parents=True)
        (job / "dense").mkdir()
        (job / "detections" / "mapping_masks").mkdir(parents=True)
        (job / "detections" / "no_cut_masks").mkdir()
        return job

    def _mask(self, path, pixel):
        array = np.zeros((6, 6), dtype=np.uint8)
        array[pixel] = 255
        Image.fromarray(array).save(path)

    def _loaders(self):
        images = {
            1: SimpleNamespace(
                name="sample.jpg",
                point3D_ids=np.asarray([10, 20]),
                xys=np.asarray([[1.0, 1.0], [4.0, 4.0]]),
            )
        }
        points = {
            10: SimpleNamespace(xyz=np.asarray([1.0, 2.0, 3.0])),
            20: SimpleNamespace(xyz=np.asarray([4.0, 5.0, 6.0])),
        }
        return lambda _: images, lambda _: points

    def test_mapper_consumes_only_approved_mapping_and_no_cut_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            mapping = job / "detections" / "mapping_masks" / "approved.png"
            no_cut = job / "detections" / "no_cut_masks" / "approved.png"
            self._mask(mapping, (1, 1))
            self._mask(no_cut, (1, 1))
            payload = {
                "policy": "yolo_only",
                "strict_research_mode": True,
                "records": [
                    {
                        "prediction_id": "approved",
                        "image_id": "sample.jpg",
                        "source": "yolo",
                        "class_id": 1,
                        "class_name": "Inclusion",
                        "confidence": 0.91,
                        "accepted_for_3d_mapping": True,
                        "accepted_for_no_cut_zone": True,
                        "mapping_mask_path": (
                            "detections/mapping_masks/approved.png"
                        ),
                        "no_cut_mask_path": "detections/no_cut_masks/approved.png",
                    },
                    {
                        "prediction_id": "visual-only",
                        "image_id": "sample.jpg",
                        "source": "opencv",
                        "class_id": None,
                        "class_name": "opencv_candidate",
                        "confidence": None,
                        "accepted_for_visualization": True,
                        "accepted_for_3d_mapping": False,
                        "accepted_for_no_cut_zone": False,
                    },
                ],
            }
            (job / "detections" / "policy_decisions.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            images_loader, points_loader = self._loaders()
            result = map_fractures_to_3d(
                job,
                images_loader=images_loader,
                points_loader=points_loader,
            )
            metadata = json.loads(
                (job / "dense" / "defect_associations.json").read_text()
            )
            visual_ply = (job / "dense" / "defects.ply").read_text()
            no_cut_ply = (job / "dense" / "no_cut_defects.ply").read_text()

        self.assertIsNotNone(result)
        self.assertEqual(metadata["mapping_point_count"], 1)
        self.assertEqual(metadata["no_cut_point_count"], 1)
        self.assertEqual(
            metadata["associations"][0]["mapping"]["prediction_ids"],
            ["approved"],
        )
        self.assertIn("element vertex 1", visual_ply)
        self.assertIn("element vertex 1", no_cut_ply)

    def test_multi_source_associations_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            records = []
            for source in ("yolo", "opencv"):
                path = (
                    job / "detections" / "mapping_masks" / f"{source}.png"
                )
                self._mask(path, (1, 1))
                records.append(
                    {
                        "prediction_id": source,
                        "image_id": "sample.jpg",
                        "source": source,
                        "class_id": 1 if source == "yolo" else None,
                        "class_name": (
                            "Inclusion"
                            if source == "yolo"
                            else "opencv_candidate"
                        ),
                        "confidence": 0.9 if source == "yolo" else None,
                        "accepted_for_3d_mapping": True,
                        "accepted_for_no_cut_zone": False,
                        "mapping_mask_path": (
                            f"detections/mapping_masks/{source}.png"
                        ),
                    }
                )
            (job / "detections" / "policy_decisions.json").write_text(
                json.dumps(
                    {
                        "policy": "union",
                        "strict_research_mode": False,
                        "records": records,
                    }
                ),
                encoding="utf-8",
            )
            images_loader, points_loader = self._loaders()
            map_fractures_to_3d(job, images_loader, points_loader)
            metadata = json.loads(
                (job / "dense" / "defect_associations.json").read_text()
            )

        association = metadata["associations"][0]["mapping"]
        self.assertEqual(association["prediction_ids"], ["opencv", "yolo"])
        self.assertEqual(association["sources"], ["opencv", "yolo"])
        self.assertEqual(association["confidences"], [0.9])
        self.assertFalse((job / "dense" / "no_cut_defects.ply").exists())

    def test_legacy_masks_without_policy_metadata_are_not_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            (job / "fractures").mkdir()
            self._mask(job / "fractures" / "legacy.png", (1, 1))
            images_loader, points_loader = self._loaders()
            result = map_fractures_to_3d(
                job,
                images_loader=images_loader,
                points_loader=points_loader,
            )
            metadata = json.loads(
                (job / "dense" / "defect_associations.json").read_text()
            )

        self.assertIsNone(result)
        self.assertEqual(metadata["status"], "unavailable")
        self.assertTrue(metadata["legacy_masks_present"])
        self.assertFalse(metadata["legacy_masks_used"])
        self.assertFalse((job / "dense" / "defects.ply").exists())


if __name__ == "__main__":
    unittest.main()
