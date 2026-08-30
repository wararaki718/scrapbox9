import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data import create_sample_data


class DataTests(unittest.TestCase):
    def test_sample_data_has_contiguous_ids_and_valid_interactions(self) -> None:
        data = create_sample_data()

        self.assertEqual([item.item_id for item in data.items], list(range(len(data.items))))
        self.assertGreaterEqual(len({item.category for item in data.items}), 3)
        self.assertTrue(all(0 <= row.user_id < data.num_users for row in data.interactions))
        self.assertTrue(all(0 <= row.item_id < len(data.items) for row in data.interactions))

if __name__ == "__main__":
    unittest.main()