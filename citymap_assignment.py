"""
===============================================================================
ASSIGNMENT: FIX THE AUTONOMOUS CAR NAVIGATION
===============================================================================

Welcome Students!

This code implements a self-driving car using Deep Q-Network (DQN) reinforcement learning.
However, several critical parameters have been intentionally set to INCORRECT values
that will prevent the car from learning properly.

YOUR TASK:
Find and fix all parameters marked with "FIX ME" comments. Use your understanding
of reinforcement learning, neural networks, and the physics of car navigation to
set appropriate values.

HINTS:
- Read the comments carefully - they explain what each parameter does
- Think about reasonable ranges (e.g., learning rates are usually 0.0001 to 0.01)
- Consider the physics (can a car sensor realistically be 1000 pixels away?)
- Test your fixes by running the program and observing the learning behavior

GRADING CRITERIA:
1. Car successfully learns to navigate (primary goal)
2. Appropriate hyperparameter values chosen
3. Understanding demonstrated in comments you add

Good luck! 🚗💨
===============================================================================
"""

import sys
import os
import math
import numpy as np
import random
from collections import deque

# --- PYTORCH ---
import torch
import torch.nn as nn
import torch.optim as optim

# --- PYQT ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsItem, QFrame, QFileDialog,
                             QTextEdit, QGridLayout)
from PyQt6.QtGui import (QImage, QPixmap, QColor, QPen, QBrush, QPainter, 
                         QPolygonF, QFont, QPainterPath)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. CONFIGURATION & THEME
# ==========================================
# Nordic Theme
C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A") 
C_ACCENT    = QColor("#88C0D0") 
C_TEXT      = QColor("#ECEFF4") 
C_SUCCESS   = QColor("#A3BE8C") 
C_FAILURE   = QColor("#BF616A") 
C_SENSOR_ON = QColor("#A3BE8C") # Green
C_SENSOR_OFF= QColor("#BF616A") # Red

# ==========================================
# PHYSICS PARAMETERS - FIX ME!
# ==========================================
CAR_WIDTH = 14     
CAR_HEIGHT = 8   
SENSOR_DIST = 3  # FIX ME! Distance sensors look ahead (pixels) - Currently unrealistic!
SENSOR_ANGLE = 90    # FIX ME! Angle spread of sensors (degrees) - Too narrow!
SPEED = 2          # FIX ME! Forward speed (pixels/step) - Way too fast!
TURN_SPEED = 15    # FIX ME! Regular turn angle (degrees/step) - Too slow!
SHARP_TURN = 45      # FIX ME! Sharp turn angle for tight corners (degrees) - Too small!

# ==========================================
# REINFORCEMENT LEARNING HYPERPARAMETERS - FIX ME!
# ==========================================
BATCH_SIZE = 128      # Number of experiences sampled per training step
GAMMA = 0.99        # Discount factor for future rewards
LR = 0.0001          # Learning rate for optimizers
TAU = 0.005         # Soft update coefficient for target networks

# TD3 SPECIFIC HYPERPARAMETERS
POLICY_NOISE = 0.2     # Noise added to target policy during critic update
NOISE_CLIP = 0.5       # Range to clip target policy noise
POLICY_DELAY = 2       # Frequency of actor & target network updates
EXPLORATION_NOISE = 0.1 # Base noise for action exploration (scaled by max_action)

MAX_CONSECUTIVE_CRASHES = 1 

EPS_DECAY_EPISODES = 250 # Cosine decay over 250 episodes

# Target Colors (for multiple targets)
TARGET_COLORS = [
    QColor(0, 255, 255),      # Cyan
    QColor(255, 100, 255),    # Magenta
    QColor(0, 255, 100),      # Green
    QColor(255, 150, 0),      # Orange
    QColor(100, 150, 255),    # Blue
    QColor(255, 50, 150),     # Pink
    QColor(150, 255, 50),     # Lime
    QColor(255, 255, 0),      # Yellow
]

# TD3: Actor Network (Predicts the best action)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Tanh restricts output to [-1, 1]
        )

    def forward(self, state):
        # In TD3 with multiple actions, we keep the network output in [-1, 1]
        # and scale it to specific ranges in the environment step function.
        return self.net(state)

# TD3: Critic Network (Dual Q-networks to reduce overestimation bias)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Critic 2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        # Combined state-action input
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        # Used for policy update (only need one Q-value)
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

# ==========================================
# 3. PHYSICS & LOGIC
# ==========================================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()
        
        # RL Init
        # TD3 Architecture setup
        # State: 7 Raycast sensors + sin(angle) + cos(angle) + norm_dist = 10 dims
        self.state_dim = 10 
        self.action_dim = 2 # Action 0: Direction, Action 1: Speed
        self.max_steer = float(SHARP_TURN)
        self.max_ray_dist = 150.0 # Maximum LIDAR-like range
        
        # Initialize Actor and its Target
        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        
        # Initialize Critic (Dual Q) and its Target
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        
        self.memory = deque(maxlen=50000)
        
        # Prioritized Replay: separate buffer for high-reward episodes
        self.priority_memory = deque(maxlen=15000)  # Store successful episodes
        self.current_episode_buffer = []  # Temporary buffer for current episode
        self.episode_scores = deque(maxlen=500)  # Track recent episode scores
        
        self.steps = 0
        self.episode_count = 0 
        self.epsilon = 1.0  
        self.min_epsilon = 0.05
        self.consecutive_crashes = 0
        
        # Locations
        self.start_pos = QPointF(100, 100) 
        self.respawn_pos = QPointF(100, 100) # Checkpoint for soft resets
        self.car_pos = QPointF(100, 100)   
        self.car_angle = 0
        self.target_pos = QPointF(200, 200)
        
        # Multiple Targets Support
        self.targets = []
        self.current_target_idx = 0
        self.targets_reached = 0
        
        self.alive = True
        self.score = 0
        self.sensor_coords = [] 
        self.prev_dist = None

    def set_start_pos(self, point):
        self.start_pos = point
        self.respawn_pos = point
        self.car_pos = point

    def reset(self):
        self.alive = True
        self.score = 0
        self.respawn_pos = QPointF(self.start_pos.x(), self.start_pos.y()) # Reset checkpoint
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)
        self.current_target_idx = 0
        self.targets_reached = 0
        if len(self.targets) > 0:
            self.target_pos = self.targets[0]
        state, dist = self.get_state()
        self.prev_dist = dist
        return state
    
    def add_target(self, point):
        self.targets.append(QPointF(point.x(), point.y()))
        if len(self.targets) == 1:
            self.target_pos = self.targets[0]
            self.current_target_idx = 0
    
    def switch_to_next_target(self):
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            self.targets_reached += 1
            return True
        return False

    def get_state(self):
        sensor_vals = []
        self.sensor_coords = []
        
        # 1. CONTINUOUS RAYCASTING (LIDAR)
        # 7 sensors: -45, -30, -15, 0, 15, 30, 45
        angles = [-45, -30, -15, 0, 15, 30, 45]
        
        for a in angles:
            rad = math.radians(self.car_angle + a)
            hit_dist = self.max_ray_dist
            
            # Sub-sample along the ray to find the nearest obstacle
            # Using 5-pixel steps for efficiency, then refining
            for d in range(5, int(self.max_ray_dist), 5):
                sx = self.car_pos.x() + math.cos(rad) * d
                sy = self.car_pos.y() + math.sin(rad) * d
                
                # Check if off-map or hitting grass
                if not (0 <= sx < self.w and 0 <= sy < self.h) or self.check_pixel(sx, sy) < 0.5:
                    hit_dist = d
                    break
            
            # Normalize hit distance to [0, 1]
            sensor_vals.append(hit_dist / self.max_ray_dist)
            
            # Store coordinates for visualization
            self.sensor_coords.append(QPointF(
                self.car_pos.x() + math.cos(rad) * hit_dist,
                self.car_pos.y() + math.sin(rad) * hit_dist
            ))
            
        # 2. TARGET ORIENTATION (Continuous sin/cos)
        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.sqrt(dx*dx + dy*dy)
        
        rad_to_target = math.atan2(dy, dx)
        angle_diff_rad = rad_to_target - math.radians(self.car_angle)
        
        # Using sin/cos removes the 180/-180 degree "jump" (discontinuity)
        target_sin = math.sin(angle_diff_rad)
        target_cos = math.cos(angle_diff_rad)
        
        norm_dist = min(dist / 1464.0, 1.0) 
        
        state = sensor_vals + [target_sin, target_cos, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def step(self, action):
        # action[0]: Direction (steering), action[1]: Speed
        # Scale [-1, 1] actions to environment ranges
        steer = action[0] * self.max_steer
        # Speed range: 0.5 to 3.0 pixels/step
        current_speed = (action[1] + 1.0) / 2.0 * (3.0 - 0.5) + 0.5
        
        self.car_angle += steer
        rad = math.radians(self.car_angle)
        
        # We can still use road color as a speed multiplier to help the agent
        speed_multiplier = 1.0
        px_x, px_y = int(self.car_pos.x()), int(self.car_pos.y())
        if 0 <= px_x < self.w and 0 <= px_y < self.h:
            c = QColor(self.map.pixel(px_x, px_y))
            # If on Red road, allow 1.5x the requested speed
            if c.red() > 230 and c.green() < 60 and c.blue() < 60:
                speed_multiplier = 1.5
            elif c.red() > 200 and c.red() > (c.green() + 30) and c.red() > (c.blue() + 30):
                speed_multiplier = 1.5

        final_speed = current_speed * speed_multiplier
        new_x = self.car_pos.x() + math.cos(rad) * final_speed
        new_y = self.car_pos.y() + math.sin(rad) * final_speed
        self.car_pos = QPointF(new_x, new_y)
        
        next_state, dist = self.get_state()
        sensors = next_state[:7]
        
        reward = -0.1
        
        # Penalize large steering to discourage aggressive wiggling
        reward -= abs(action[0]) * 0.2
            
        done = False
        
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())
        
        if car_center_val < 0.4:
            reward = -100
            done = True
            self.alive = False
        elif dist < 20: 
            reward += 100
            # Checkpoint Reached! Update respawn position to here.
            self.respawn_pos = QPointF(self.car_pos.x(), self.car_pos.y())
            has_next = self.switch_to_next_target()
            if has_next:
                done = False
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
            else:
                done = True
        else:
            # 1. HEADING REWARD: target_cos (index 8) is 1.0 when facing target, -1.0 when facing away
            target_sin = next_state[7]
            target_cos = next_state[8]
            
            # Reward for facing the target (even if not moving)
            reward += target_cos * 0.5 
            
            # 2. PROGRESS REWARD: Speed toward the target
            if self.prev_dist is not None:
                dist_moved = self.prev_dist - dist
                # Significant reward for closing distance
                if dist_moved > 0:
                    # Closing distance + facing target = Good progress
                    # We multiply by target_cos so moving "sideways" near the target is less rewarded
                    reward += dist_moved * 3.0 * max(0, target_cos)
                else:
                    # Penalize moving away from the target
                    reward += dist_moved * 2.0 
            
            self.prev_dist = dist
            
        self.score += reward
        return next_state, reward, done

    def check_pixel(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            
            # 1. SAFE: WHITE (High brightness) matched to 255,255,255
            if c.red() > 240 and c.green() > 240 and c.blue() > 240:
                return 1.0  
            
            # 2. SAFE: RED (High speed roads) matched to 240,43,24
            if c.red() > 230 and c.green() < 60 and c.blue() < 60:
                return 1.0

            # 3. SAFE: BLENDED EDGES (Transition between Red and White)
            # Both Red and White have high Red channel. The anti-aliased pixels between them
            # might fail strict Red or strict White checks, but they are still road.
            if c.red() > 200:
                return 1.0
                
            # # 3. CRASH: GREEN (Grass/Terrain)
            # # Use a margin of 10 to avoid crashing on noisy grey/white pixels
            # # where Green might be just slightly higher than Red/Blue
            # if c.green() > (c.red() + 10) and c.green() > (c.blue() + 10):
            #     return 0.0
                
            # return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

    def optimize(self):
        total_memory_size = len(self.memory) + len(self.priority_memory)
        if total_memory_size < BATCH_SIZE: return 0
        
        # --- 1. SAMPLING ---
        success_rate = len(self.priority_memory) / max(total_memory_size, 1)
        priority_ratio = 0.3 + (success_rate * 0.4)
        priority_samples = int(BATCH_SIZE * priority_ratio)
        regular_samples = BATCH_SIZE - priority_samples
        
        batch = []
        if len(self.priority_memory) >= priority_samples:
            batch.extend(random.sample(self.priority_memory, priority_samples))
        else:
            batch.extend(list(self.priority_memory))
            regular_samples += priority_samples - len(self.priority_memory)
        
        if len(self.memory) >= regular_samples:
            batch.extend(random.sample(self.memory, regular_samples))
        else:
            batch.extend(list(self.memory))
        
        if len(batch) < BATCH_SIZE // 2: return 0
        
        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s))
        a = torch.FloatTensor(np.array(a)) # Action is now (Batch, 2)
        r = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(np.array(ns))
        d = torch.FloatTensor(d).unsqueeze(1)

        # --- TD3: 2. TWIN CRITIC UPDATE ---
        with torch.no_grad():
            # TD3 Target Policy Smoothing:
            # Add noise to target action to smooth out Q-values
            # Actions are in [-1, 1], so we clamp noise to stay within range
            noise = (torch.randn_like(a) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(ns) + noise).clamp(-1.0, 1.0)

            # TD3 Clipped Double-Q Learning:
            # Take minimum of two target Q-values to reduce overestimation
            target_Q1, target_Q2 = self.critic_target(ns, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (GAMMA * target_Q * (1 - d))

        # Current Q estimates
        current_Q1, current_Q2 = self.critic(s, a)
        
        # Compute loss for both critics
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- TD3: 3. DELAYED POLICY UPDATE ---
        # Update Actor and Target networks only every POLICY_DELAY steps
        actor_loss = None
        if self.steps % POLICY_DELAY == 0:
            # Compute actor loss (maximize Q-value)
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return critic_loss.item()
    
    def store_experience(self, experience):
        self.current_episode_buffer.append(experience)
    
    def finalize_episode(self, episode_reward):
        self.episode_count += 1
        
        # Episode-based Cosine Decay
        if self.episode_count < EPS_DECAY_EPISODES:
            frac = self.episode_count / EPS_DECAY_EPISODES
            self.epsilon = self.min_epsilon + 0.5 * (1.0 - self.min_epsilon) * (1 + math.cos(math.pi * frac))
        else:
            self.epsilon = self.min_epsilon

        if len(self.current_episode_buffer) == 0:
            return
        
        self.episode_scores.append(episode_reward)
        
        if not self.alive:
            self.consecutive_crashes += 1
        else:
            self.consecutive_crashes = 0
        
        if episode_reward > 0:
            for exp in self.current_episode_buffer:
                self.priority_memory.append(exp)
        else:
            for exp in self.current_episode_buffer:
                self.memory.append(exp)
        
        self.current_episode_buffer = []

# ==========================================
# 4. CUSTOM WIDGETS (VISUALS)
# ==========================================
class LineChart(QWidget):
    def __init__(self, title="Chart", color=C_ACCENT, height=120):
        super().__init__()
        self.setMinimumHeight(height)
        self.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        self.data_points = []
        self.max_points = 100
        self.title = title
        self.line_color = color

    def update_chart(self, new_val):
        self.data_points.append(new_val)
        if len(self.data_points) > self.max_points:
            self.data_points.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, C_PANEL)
        
        # Title
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(10, 15, self.title)
        
        if len(self.data_points) < 2:
            return

        min_val = min(self.data_points)
        max_val = max(self.data_points)
        
        if max_val == min_val: max_val += 0.001
        
        points = []
        step_x = w / (self.max_points - 1)
        
        for i, val in enumerate(self.data_points):
            x = i * step_x
            ratio = (val - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.7) + (h * 0.1))
            points.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)
            
        pen = QPen(self.line_color, 2)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Current Value text
        current_val = self.data_points[-1]
        painter.drawText(w - 50, 15, f"{current_val:.2f}")

class SensorItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.pulse = 0
        self.pulse_speed = 0.3
        self.is_detecting = True
        
    def set_detecting(self, detecting):
        self.is_detecting = detecting
        self.update()
    
    def boundingRect(self):
        return QRectF(-4, -4, 8, 8)
    
    def paint(self, painter, option, widget):
        self.pulse += self.pulse_speed
        if self.pulse > 1.0:
            self.pulse = 0
        
        if self.is_detecting:
            color = C_SENSOR_ON
            outer_alpha = int(150 * (1 - self.pulse))
        else:
            color = C_SENSOR_OFF
            outer_alpha = int(200 * (1 - self.pulse))
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        outer_size = 3 + (2 * self.pulse)
        outer_color = QColor(color)
        outer_color.setAlpha(outer_alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outer_color))
        painter.drawEllipse(QPointF(0, 0), outer_size, outer_size)
        
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 2, 2)

class CarItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        self.brush = QBrush(C_ACCENT)
        self.pen = QPen(Qt.GlobalColor.white, 1)

    def boundingRect(self):
        return QRectF(-CAR_WIDTH/2, -CAR_HEIGHT/2, CAR_WIDTH, CAR_HEIGHT)

    def paint(self, painter, option, widget):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(CAR_WIDTH/2)-2, -3, 2, 6)

class TargetItem(QGraphicsItem):
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(50)
        self.pulse = 0
        self.growing = True
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active
        self.number = number

    def set_active(self, active):
        self.is_active = active
        self.update()
    
    def set_color(self, color):
        self.color = color
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget):
        if self.is_active:
            if self.growing:
                self.pulse += 0.5
                if self.pulse > 10: self.growing = False
            else:
                self.pulse -= 0.5
                if self.pulse < 0: self.growing = True
            
            r = 10 + self.pulse
            painter.setPen(Qt.PenStyle.NoPen)
            outer_color = QColor(self.color)
            outer_color.setAlpha(100)
            painter.setBrush(QBrush(outer_color)) 
            painter.drawEllipse(QPointF(0,0), r, r)
            painter.setBrush(QBrush(self.color)) 
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPointF(0,0), 8, 8)
        else:
            dimmed_color = QColor(self.color)
            dimmed_color.setAlpha(120)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(QBrush(dimmed_color))
            painter.drawEllipse(QPointF(0,0), 6, 6)
        
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(-10, -10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(self.number))

# ==========================================
# 5. APP
# ==========================================
class NeuralNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralNav: ASSIGNMENT - Fix the Parameters!")
        self.resize(1300, 850)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C_BG_DARK.name()}; }}
            QLabel {{ color: {C_TEXT.name()}; font-family: Segoe UI; font-size: 13px; }}
            QPushButton {{ background-color: {C_PANEL.name()}; color: white; border: 1px solid {C_INFO_BG.name()}; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {C_INFO_BG.name()}; }}
            QPushButton:checked {{ background-color: {C_ACCENT.name()}; color: black; }}
            QTextEdit {{ background-color: {C_PANEL.name()}; color: #D8DEE9; border: none; font-family: Consolas; font-size: 11px; }}
            QFrame {{ border: none; }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT PANEL
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(10)
        
        lbl_title = QLabel("CONTROLS")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        vbox.addWidget(lbl_title)
        
        self.lbl_status = QLabel("1. Click Map -> CAR\n2. Click Map -> TARGET(S)\n   (Multiple clicks for sequence)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px; color: #E5E9F0;")
        vbox.addWidget(self.lbl_status)

        self.btn_run = QPushButton("▶ START (Space)")
        self.btn_run.setCheckable(True)
        self.btn_run.setEnabled(False) 
        self.btn_run.clicked.connect(self.toggle_training)
        vbox.addWidget(self.btn_run)
        
        self.btn_reset = QPushButton("↺ RESET ALL")
        self.btn_reset.clicked.connect(self.full_reset)
        vbox.addWidget(self.btn_reset)
        
        self.btn_load = QPushButton("📂 LOAD MAP")
        self.btn_load.clicked.connect(self.load_map_dialog)
        vbox.addWidget(self.btn_load)

        vbox.addSpacing(15)
        vbox.addSpacing(15)
        vbox.addWidget(QLabel("METRICS"))
        
        self.chart_rewards = LineChart("Rewards", C_SUCCESS, height=100)
        vbox.addWidget(self.chart_rewards)
        
        self.chart_loss = LineChart("Loss", C_FAILURE, height=100)
        vbox.addWidget(self.chart_loss)
        
        self.chart_eps = LineChart("Epsilon", C_ACCENT, height=80)
        vbox.addWidget(self.chart_eps)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 10, 10, 10)
        
        self.val_eps = QLabel("1.00")
        self.val_eps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Epsilon:"), 0,0)
        sf_layout.addWidget(self.val_eps, 0,1)
        
        self.val_rew = QLabel("0")
        self.val_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Last Reward:"), 1,0)
        sf_layout.addWidget(self.val_rew, 1,1)
        
        vbox.addWidget(stats_frame)

        vbox.addWidget(QLabel("LOGS"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # RIGHT PANEL
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)

        # Logic
        self.setup_map("city_map.png") 
        self.setup_state = 0 
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.game_loop)
        
        self.car_item = CarItem()
        self.target_items = []
        self.sensor_items = []
        # Support for 7 sensors
        for _ in range(7):
            si = SensorItem()
            self.scene.addItem(si)
            self.sensor_items.append(si)

    def log(self, msg):
        self.log_console.append(msg)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def setup_map(self, path):
        if not os.path.exists(path):
            self.create_dummy_map(path)
        self.map_img = QImage(path).convertToFormat(QImage.Format.Format_RGB32)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self.brain = CarBrain(self.map_img)
        self.log(f"Map Loaded.")

    def create_dummy_map(self, path):
        img = QImage(1000, 800, QImage.Format.Format_RGB32)
        img.fill(C_BG_DARK)
        p = QPainter(img)
        p.setBrush(Qt.GlobalColor.white)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(100, 100, 800, 600)
        p.setBrush(C_BG_DARK)
        p.drawEllipse(250, 250, 500, 300)
        p.end()
        img.save(path)

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg)")
        if f: 
            self.full_reset()
            self.setup_map(f)

    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())
        if self.setup_state == 0:
            self.brain.set_start_pos(pt) 
            self.scene.addItem(self.car_item)
            self.car_item.setPos(pt)
            self.setup_state = 1
            self.lbl_status.setText("Click Map -> TARGET(S)\nRight-click when done")
        elif self.setup_state == 1:
            if event.button() == Qt.MouseButton.LeftButton:
                self.brain.add_target(pt)
                target_idx = len(self.brain.targets) - 1
                color = TARGET_COLORS[target_idx % len(TARGET_COLORS)]
                is_active = (target_idx == 0)
                num_targets = len(self.brain.targets)
                
                target_item = TargetItem(color, is_active, num_targets)
                target_item.setPos(pt)
                self.scene.addItem(target_item)
                self.target_items.append(target_item)
                
                self.lbl_status.setText(f"Targets: {num_targets}\nRight-click to finish setup")
                self.log(f"Target #{num_targets} added at ({pt.x():.0f}, {pt.y():.0f})")
            
            elif event.button() == Qt.MouseButton.RightButton:
                if len(self.brain.targets) > 0:
                    self.setup_state = 2
                    self.lbl_status.setText(f"READY. {len(self.brain.targets)} target(s). Press SPACE.")
                    self.lbl_status.setStyleSheet(f"background-color: {C_SUCCESS.name()}; color: #2E3440; font-weight: bold; padding: 10px; border-radius: 5px;")
                    self.btn_run.setEnabled(True)
                    self.update_visuals()

    def full_reset(self):
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setEnabled(False)
        self.setup_state = 0
        self.scene.removeItem(self.car_item)
        for target_item in self.target_items:
            self.scene.removeItem(target_item)
        self.target_items = []
        self.brain.targets = []
        self.brain.current_target_idx = 0
        self.brain.targets_reached = 0
        
        for s in self.sensor_items: 
            if s.scene() == self.scene: self.scene.removeItem(s)
        self.lbl_status.setText("1. Click Map -> CAR\n2. Click Map -> TARGET(S)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; color: white; padding: 10px; border-radius: 5px;")
        self.log("--- RESET ---")
        self.chart_rewards.data_points = []
        self.chart_rewards.update()
        self.chart_loss.data_points = []
        self.chart_loss.update()
        self.chart_eps.data_points = []
        self.chart_eps.update()

    def toggle_training(self):
        if self.btn_run.isChecked():
            self.sim_timer.start(16)
            self.btn_run.setText("⏸ PAUSE")
        else:
            self.sim_timer.stop()
            self.btn_run.setText("▶ RESUME")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.setup_state == 2:
            self.btn_run.click()

    def game_loop(self):
        if self.setup_state != 2: return

        state, _ = self.brain.get_state()
        action = 0
        
        prev_target_idx = self.brain.current_target_idx
        
        # TD3 Action Selection with Exploration Noise
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # Actor outputs 2 actions in [-1, 1] range
            action_orig = self.brain.actor(state_tensor).cpu().data.numpy().flatten()
            
            # Add Gaussian exploration noise to both actions
            noise = np.random.normal(0, EXPLORATION_NOISE, size=self.brain.action_dim)
            # Scale noise by epsilon for decay
            action = np.clip(action_orig + noise * self.brain.epsilon, -1.0, 1.0)

        next_s, rew, done = self.brain.step(action)
        
        self.brain.store_experience((state, action, rew, next_s, done))
        loss = self.brain.optimize()
        if loss and self.brain.steps % 50 == 0:
            # Update Loss Chart (downsampled)
            self.chart_loss.update_chart(loss)
            # Update Epsilon Chart
            self.chart_eps.update_chart(self.brain.epsilon)
            
        if loss and self.brain.steps % 100 == 0:
            self.log(f"Step {self.brain.steps}: Loss = {loss:.5f} | Epsilon = {self.brain.epsilon:.3f}")
        
        if self.brain.current_target_idx != prev_target_idx:
            target_num = self.brain.current_target_idx + 1
            total = len(self.brain.targets)
            self.log(f"<font color='#88C0D0'>🎯 Target {prev_target_idx + 1} reached! Moving to target {target_num}/{total}</font>")
        
        # Target updates are now handled inside brain.optimize() for TD3
        
        self.brain.steps += 1
        
        if done:
            self.brain.finalize_episode(self.brain.score)
            
            should_reset_position = False
            
            if self.brain.consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                self.log(f"<font color='#BF616A'><b>⚠️ {MAX_CONSECUTIVE_CRASHES} consecutive crashes! Resetting to origin...</b></font>")
                self.log(f"<font color='#88C0D0'>💡 Tip: Adjust hyperparameters, simplify map, or increase exploration (epsilon)</font>")
                self.brain.consecutive_crashes = 0
                should_reset_position = True
            
            if not self.brain.alive:
                txt = f"CRASH ({self.brain.consecutive_crashes}/{MAX_CONSECUTIVE_CRASHES})"
                col = "#BF616A"
            else:
                if self.brain.targets_reached == len(self.brain.targets) - 1:
                    txt = f"ALL {len(self.brain.targets)} TARGETS COMPLETED!"
                    col = "#A3BE8C"
                else:
                    txt = "GOAL"
                    col = "#A3BE8C"
                should_reset_position = True
            
            priority_size = len(self.brain.priority_memory)
            regular_size = len(self.brain.memory)
            total_mem = priority_size + regular_size
            priority_pct = (priority_size / total_mem * 100) if total_mem > 0 else 0
            
            success_rate = priority_size / max(total_mem, 1)
            sampling_ratio = 0.3 + (success_rate * 0.4)
            
            avg_score = sum(self.brain.episode_scores) / len(self.brain.episode_scores) if self.brain.episode_scores else 0
            
            self.log(f"<font color='{col}'>Ep: {self.brain.episode_count} | {txt} | Targets: {self.brain.targets_reached} | Scr: {self.brain.score:.0f} (Avg: {avg_score:.1f}) | "
                    f"Mem: {priority_size}P/{regular_size}R ({priority_pct:.1f}%) | "
                    f"Sample: {sampling_ratio*100:.0f}%P</font>")
            
            self.chart_rewards.update_chart(self.brain.score)
            
            if should_reset_position:
                self.brain.reset()
            else:
                self.brain.score = 0
                self.brain.alive = True
                
                # SOFT RESET: Respawn at last checkpoint (Start or Last Target)
                self.brain.car_pos = QPointF(self.brain.respawn_pos.x(), self.brain.respawn_pos.y())
                self.brain.car_angle = random.randint(0, 360) # Randomize direction to avoid loop stuckness
                
                # Do NOT reset targets here. Keep fighting for the current target!
                # If we are stuck, consecutive_crashes will eventually trigger a full reset.
                _, dist = self.brain.get_state()
                self.brain.prev_dist = dist

        self.update_visuals()
        self.val_eps.setText(f"{self.brain.epsilon:.3f}")
        self.val_rew.setText(f"{self.brain.score:.0f}")

    def update_visuals(self):
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)
        
        for i, target_item in enumerate(self.target_items):
            is_active = (i == self.brain.current_target_idx)
            target_item.set_active(is_active)
        
        self.scene.update() 
        
        for i, coord in enumerate(self.brain.sensor_coords):
            self.sensor_items[i].setPos(coord)
            s_val = self.brain.get_state()[0][i]
            # If hit distance is less than max, it's detecting something
            self.sensor_items[i].set_detecting(s_val < 0.95)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NeuralNavApp()
    win.show()
    sys.exit(app.exec())

