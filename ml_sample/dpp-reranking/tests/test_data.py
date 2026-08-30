import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data import create_reranking_data, create_training_data


class DataTests(unittest.TestCase):
    def test_training_and_reranking_data_share_catalog_but_not_interactions(self) -> None:
        training_data = create_training_data()
        reranking_data = create_reranking_data()

        self.assertEqual(training_data.items, reranking_data.items)
        self.assertEqual(len(training_data.items), 30)
        self.assertNotEqual(training_data.interactions, reranking_data.interactions)
        known_item_ids = {
            row.item_id for row in reranking_data.interactions if row.user_id == 0
        }
        self.assertEqual(len(reranking_data.items) - len(known_item_ids), 27)

        for data in (training_data, reranking_data):
            with self.subTest(data=data):
                self.assertEqual(
                    [item.item_id for item in data.items], list(range(len(data.items)))
                )
                self.assertGreaterEqual(len({item.category for item in data.items}), 3)
                self.assertTrue(
                    all(0 <= row.user_id < data.num_users for row in data.interactions)
                )
                self.assertTrue(
                    all(0 <= row.item_id < len(data.items) for row in data.interactions)
                )

if __name__ == "__main__":
    unittest.main()