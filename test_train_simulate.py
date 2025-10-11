import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from dqn_inverted_pendulum import InvertedPendulumDQN

class TestTrainSimulate(unittest.TestCase):
    
    def setUp(self):
        self.dqn = InvertedPendulumDQN()
    
    def test_train_returns_history(self):
        history = self.dqn.train(episodes=2)
        self.assertEqual(len(history), 2)
        self.assertIsInstance(history[0], tuple)
        self.assertEqual(len(history[0]), 2)
    
    def test_train_epsilon_decay(self):
        initial_epsilon = self.dqn.epsilon
        self.dqn.train(episodes=11)
        self.assertLess(self.dqn.epsilon, initial_epsilon)
    
    def test_train_q_table_updates(self):
        initial_sum = self.dqn.q_table.sum().sum()
        self.dqn.train(episodes=5)
        final_sum = self.dqn.q_table.sum().sum()
        self.assertNotEqual(initial_sum, final_sum)
    
    @patch('matplotlib.pyplot.show')
    @patch('dqn_inverted_pendulum.FuncAnimation')
    def test_simulate_trained(self, mock_animation, mock_show):
        mock_animation.return_value = MagicMock()
        result = self.dqn.simulate_trained()
        mock_animation.assert_called_once()
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()