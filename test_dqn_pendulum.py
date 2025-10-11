import unittest
import numpy as np
from dqn_inverted_pendulum import InvertedPendulumDQN

class TestInvertedPendulumDQN(unittest.TestCase):
    
    def setUp(self):
        self.dqn = InvertedPendulumDQN()
    
    def test_discretize_state(self):
        # Test normal state
        state = [0.0, 0.0, 0.0, 0.0]
        idx = self.dqn.discretize_state(state)
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)
        
        # Test boundary states
        state = [5.0, 10.0, np.pi, 10.0]
        idx = self.dqn.discretize_state(state)
        self.assertIsInstance(idx, int)
    
    def test_get_reward(self):
        # Test upright position (should be positive)
        state = [0.0, 0.0, 0.05, 0.0]
        reward = self.dqn.get_reward(state)
        self.assertGreater(reward, 0)
        
        # Test fallen position (should be negative)
        state = [0.0, 0.0, np.pi/2 + 0.1, 0.0]
        reward = self.dqn.get_reward(state)
        self.assertEqual(reward, -100)
    
    def test_choose_action(self):
        action = self.dqn.choose_action(0)
        self.assertIn(action, range(len(self.dqn.actions)))
    
    def test_dynamics(self):
        state = [0.0, 0.0, 0.1, 0.0]
        derivatives = self.dqn.dynamics(0, state, 0)
        self.assertEqual(len(derivatives), 4)
        self.assertIsInstance(derivatives[0], (int, float))

if __name__ == '__main__':
    unittest.main()