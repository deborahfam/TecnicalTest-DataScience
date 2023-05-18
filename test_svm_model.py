import unittest
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from svm_model import SVM_model

class TestSVMModel(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {'Number': [1, 2, 3, 4, 5],
                'Label': ['None', 'None', 'Fizz', 'None', 'Buzz']}
        self.df = pd.DataFrame(data)

    def test_SVM_model(self):
        # Test the SVM_model function
        accuracy = SVM_model(self.df)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_SVM_model_binary_classification(self):
        # Test with a DataFrame for binary classification
        binary_df = pd.DataFrame({'Number': [1, 2, 3, 4, 5],
                                  'Label': ['None', 'None', 'Fizz', 'None', 'Buzz']})
        accuracy = SVM_model(binary_df)
        self.assertEqual(accuracy, 1.0)

    def test_SVM_model_multiclass_classification(self):
        # Test with a DataFrame for multiclass classification
        multiclass_df = pd.DataFrame({'Number': [1, 2, 3, 5, 15],
                                      'Label': ['None', 'None', 'Fizz', 'Buzz', 'FizzBuzz']})
        accuracy = SVM_model(multiclass_df)
        self.assertEqual(accuracy, 1.0)

    def test_SVM_model_high_accuracy(self):
        # Test with a DataFrame that should achieve high accuracy
        high_accuracy_df = pd.DataFrame({'Number': [3, 6, 9, 12, 15],
                                         'Label': ['Fizz', 'Fizz', 'Fizz', 'Fizz', 'FizzBuzz']})
        accuracy = SVM_model(high_accuracy_df)
        self.assertEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()

