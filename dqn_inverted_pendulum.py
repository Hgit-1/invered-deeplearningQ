import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import pickle
import os

class InvertedPendulumDQN:
    def __init__(self):
        # DQN parameters
        self.epsilon = 0.5
        self.alpha = 0.1
        self.gamma = 0.9
        
        # Pendulum parameters
        self.M = 1.0   # cart mass
        self.m = 0.1   # pole mass
        self.l = 1.0   # pole length
        self.g = 9.81  # gravity
        self.b = 0.05  # cart friction
        
        # Discretized state space
        self.angle_bins = 20
        self.velocity_bins = 20
        self.position_bins = 20
        self.ang_vel_bins = 20
        
        # Action space: force values
        self.actions = [-20, -10, -5, 0, 5, 10, 20]
        
        # Initialize Q-table
        total_states = self.angle_bins * self.velocity_bins * self.position_bins * self.ang_vel_bins
        self.q_table = np.zeros((total_states, len(self.actions)))
        
        # State bounds for discretization
        self.angle_bounds = (-np.pi, np.pi)
        self.velocity_bounds = (-10, 10)
        self.position_bounds = (-100, 100)
        self.ang_vel_bounds = (-10, 10)
        
    def discretize_state(self, state):
        x, v, theta, omega = state
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        bounds = [self.position_bounds, self.velocity_bounds, self.angle_bounds, self.ang_vel_bounds]
        bins = [self.position_bins, self.velocity_bins, self.angle_bins, self.ang_vel_bins]
        
        indices = []
        for val, (low, high), n_bins in zip([x, v, theta, omega], bounds, bins):
            idx = np.clip(int((val - low) / (high - low) * n_bins), 0, n_bins - 1)
            indices.append(idx)
        
        return np.ravel_multi_index(indices, bins)
    
    def get_reward(self, state):
        x, v, theta, omega = state
        
        # Normalize angle
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # Reward for keeping pole upright and cart centered
        angle_reward = -abs(theta) * 10
        position_reward = -abs(x)
        velocity_penalty = -0.1 * (v**2 + omega**2)
        
        # Bonus for being very close to target
        if abs(theta) < 0.1 and abs(x) < 0.5:
            return 100 + angle_reward + position_reward + velocity_penalty
        
        # Penalty for falling or hitting walls
        if abs(theta) > np.pi/2 or abs(x) > 100:
            return -100
            
        return angle_reward + position_reward + velocity_penalty
    
    def dynamics(self, t, s, F):
        x, v, th, w = s
        I = (1/3) * self.m * (2 * self.l)**2
        
        def equations(vars):
            xddot, wdot = vars
            
            x_gc_ddot = xddot + self.l * wdot * np.cos(th) - self.l * w**2 * np.sin(th)
            y_gc_ddot = self.l * wdot * np.sin(th) + self.l * w**2 * np.cos(th)
            
            N = self.m * x_gc_ddot
            P = self.m * self.g + self.m * y_gc_ddot
            
            eq1 = F - self.M * xddot - N - self.b * v
            eq2 = N * np.cos(th) - P * np.sin(th) - I * wdot / (2 * self.l)
            
            return [eq1, eq2]
        
        from scipy.optimize import fsolve
        xddot, wdot = fsolve(equations, [0, 0])
        
        return [v, xddot, w, wdot]
    
    def choose_action(self, state_idx):
        if random.uniform(0, 1) < self.epsilon or np.all(self.q_table[state_idx] == 0):
            return random.randint(0, len(self.actions) - 1)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def save_model(self, filename='dqn_model.pkl'):
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'parameters': {
                'alpha': self.alpha, 'gamma': self.gamma,
                'angle_bins': self.angle_bins, 'velocity_bins': self.velocity_bins,
                'position_bins': self.position_bins, 'ang_vel_bins': self.ang_vel_bins
            }
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_model(self, filename='dqn_model.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            return True
        return False
    
    def train(self, episodes=100, save_interval=50):
        history = []
        
        for episode in range(episodes):
            # Reset environment
            state = np.array([0.0, 0.0, 0.1, 0.0])  # small initial angle
            total_reward = 0
            steps = 0
            max_steps = 500
            
            while steps < max_steps:
                current_state_idx = self.discretize_state(state)
                action_idx = self.choose_action(current_state_idx)
                force = self.actions[action_idx]
                
                # Simulate one step
                sol = solve_ivp(lambda t, y: self.dynamics(t, y, force), 
                               [0, 0.02], state, t_eval=[0.02])
                next_state = sol.y[:, -1]
                
                # Apply wall constraints
                next_state[0] = np.clip(next_state[0], -100, 100)
                if abs(next_state[0]) >= 100:
                    next_state[1] = 0  # Stop velocity at wall
                
                reward = self.get_reward(next_state)
                next_state_idx = self.discretize_state(next_state)
                
                # Q-learning update
                max_next_q = np.max(self.q_table[next_state_idx])
                self.q_table[current_state_idx, action_idx] += self.alpha * (
                    reward + self.gamma * max_next_q - self.q_table[current_state_idx, action_idx]
                )
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Check if episode should end
                if abs(state[2]) > np.pi/2 or abs(state[0]) >= 100:
                    break
            
            history.append((steps, total_reward))
            
            # Decay epsilon
            if episode % 10 == 0:
                self.epsilon = max(0.1, self.epsilon * 0.95)
                
            print(f'Episode {episode}: steps={steps}, reward={total_reward:.2f}, epsilon={self.epsilon:.3f}')
            
            if (episode + 1) % save_interval == 0:
                self.save_model()
                print(f'Model saved at episode {episode + 1}')
        
        self.save_model()
        return history
    
    def simulate_trained(self):
        # Test the trained agent
        state = np.array([0.0, 0.0, 0.2, 0.0])  # larger initial angle
        
        # Setup animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pendulum visualization
        ax1.set_xlim(-100, 100)
        ax1.set_ylim(-20, 20)
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.set_title('DQN Controlled Inverted Pendulum')
        
        cartW, cartH = 0.4, 0.2
        cart = plt.Rectangle((-cartW/2, 0), cartW, cartH, facecolor=(0, 0.6, 0.8))
        rod, = ax1.plot([], [], lw=3, color='r')
        ax1.add_patch(cart)
        
        # Performance plots
        ax2.set_title('Performance Metrics')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Angle (rad)')
        
        angles = []
        forces = []
        
        def update(frame):
            nonlocal state
            
            if frame < 1000:  # Run for 1000 steps
                state_idx = self.discretize_state(state)
                action_idx = np.argmax(self.q_table[state_idx])
                force = self.actions[action_idx]
                
                sol = solve_ivp(lambda t, y: self.dynamics(t, y, force), 
                               [0, 0.02], state, t_eval=[0.02])
                state = sol.y[:, -1]
                
                # Apply wall constraints
                state[0] = np.clip(state[0], -100, 100)
                if abs(state[0]) >= 100:
                    state[1] = 0
                
                angles.append(state[2])
                forces.append(force)
                
                # Update pendulum
                x, theta = state[0], state[2]
                cart.set_xy((x - cartW/2, 0))
                
                rod_x = [x, x + 2 * self.l * np.sin(theta)]
                rod_y = [cartH, cartH - 2 * self.l * np.cos(theta)]
                rod.set_data(rod_x, rod_y)
                
                # Update plots
                if len(angles) > 1:
                    ax2.clear()
                    ax2.plot(angles, label='Angle')
                    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    ax2.set_title('Angle Control')
                    ax2.legend()
            
            return cart, rod
        
        ani = FuncAnimation(fig, update, frames=1000, interval=20, blit=False)
        plt.tight_layout()
        plt.show()
        
        return ani

# Usage
if __name__ == "__main__":
    dqn_pendulum = InvertedPendulumDQN()
    
    # Try to load existing model
    if dqn_pendulum.load_model():
        print("Loaded existing model. Continuing training...")
    else:
        print("No existing model found. Starting fresh training...")
    
    history = dqn_pendulum.train(episodes=100)
    
    print("\nSimulating trained agent...")
    animation = dqn_pendulum.simulate_trained()