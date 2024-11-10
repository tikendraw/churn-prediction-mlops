import unittest

import pandas as pd

from src.data_validation import Customer, schema_validation


class TestSchemaValidation(unittest.TestCase):

    def test_valid_schema(self):
        """
        Test that schema_validation returns True for a valid CSV file.
        """
        self.assertTrue(schema_validation("dataset/sample_data.csv"))

    def test_invalid_schema_wrong_data_type(self):
        """
        Test that schema_validation returns False for a CSV file with an incorrect data type.
        """
        # Read the sample data
        df = pd.read_csv("dataset/sample_data.csv")
        
        # Intentionally change the data type of a column
        df['Age'] = df['Age'].astype(str)
        
        # Save the modified DataFrame to a temporary CSV file
        df.to_csv("dataset/temp.csv", index=False)

        # Test with the modified CSV
        self.assertFalse(schema_validation("dataset/temp.csv"))

if __name__ == '__main__':
    unittest.main()
