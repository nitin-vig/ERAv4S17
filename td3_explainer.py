import sys
import torch
import torch.nn as nn
import numpy as np
import math
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame)
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QPolygonF
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. VISUAL CANVAS
# ==========================================

class TD3FlowCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.phase = "IDLE"
        self.show_grads = False
        self.show_sync = False
        self.data_packet = None # (s, a, r, ns, d)
        
    def set_state(self, phase, data=None, show_grads=False, show_sync=False):
        self.phase = phase
        self.data_packet = data
        self.show_grads = show_grads
        self.show_sync = show_sync
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#242933"))
        
        # --- DEFINE NODE POSITIONS ---
        actor_x = w * 0.4
        actor_y = h * 0.2
        critic_x = w * 0.4
        critic_y = h * 0.5
        
        data_x = w * 0.1
        target_offset = 180
        loss_x = w * 0.85

        l_actor = QPointF(actor_x, actor_y)
        t_actor = QPointF(actor_x + target_offset, actor_y)
        l_critic = QPointF(critic_x, critic_y)
        t_critic = QPointF(critic_x + target_offset, critic_y)
        
        env_node = QPointF(data_x, h * 0.35)
        buffer_node = QPointF(data_x, h * 0.65)
        loss_node = QPointF(loss_x, h * 0.5)

        # Draw Group Boxes
        self.draw_group_box(painter, l_actor, t_actor, "ACTOR (Policy Control)", "#5E81AC")
        self.draw_group_box(painter, l_critic, t_critic, "CRITIC (Value Judgment)", "#BF616A")

        # --- DRAW PHASE CONNECTIONS ---
        painter.setPen(QPen(QColor("#4C566A"), 2))
        
        if self.phase == "COLLECT":
            self.draw_arrow(painter, l_actor, env_node)
            self.draw_arrow(painter, env_node, buffer_node)
            self.draw_data_tag(painter, self.lerp(l_actor, env_node, 0.5), "Action (a)")
            self.draw_data_tag(painter, self.lerp(env_node, buffer_node, 0.5), "Tuple (s, a, r, s')")

        elif self.phase == "CRITIC_LEARN":
            # Show data being pulled from buffer
            self.draw_arrow(painter, buffer_node, l_critic)
            self.draw_arrow(painter, buffer_node, t_actor) # s' goes to target actor
            self.draw_arrow(painter, t_actor, t_critic) # next_a goes to target critic
            self.draw_arrow(painter, t_critic, loss_node)
            self.draw_arrow(painter, l_critic, loss_node)
            
            if self.show_grads:
                grad_pen = QPen(QColor("#BF616A"), 3, Qt.PenStyle.DashLine)
                painter.setPen(grad_pen)
                self.draw_arrow(painter, loss_node, l_critic)

        elif self.phase == "ACTOR_LEARN":
            self.draw_arrow(painter, buffer_node, l_actor)
            self.draw_arrow(painter, l_actor, l_critic)
            
            if self.show_grads:
                grad_pen = QPen(QColor("#BF616A"), 3, Qt.PenStyle.DashLine)
                painter.setPen(grad_pen)
                self.draw_arrow(painter, l_critic, l_actor)

        # --- DRAW NODES ---
        self.draw_node(painter, l_actor, "LIVE\nACTOR", "#88C0D0")
        self.draw_node(painter, t_actor, "TARGET\nACTOR", "#88C0D0", 140)
        self.draw_node(painter, l_critic, "LIVE\nCRITICS", "#BF616A")
        self.draw_node(painter, t_critic, "TARGET\nCRITICS", "#BF616A", 140)
        self.draw_node(painter, env_node, "CITY\nMAP", "#A3BE8C")
        self.draw_node(painter, buffer_node, "REPLAY\nBUFFER", "#EBCB8B")
        self.draw_node(painter, loss_node, "MSE\nLOSS", "#D08770")

    def draw_group_box(self, painter, p1, p2, label, color):
        rect = QRectF(p1.x() - 80, p1.y() - 50, (p2.x() - p1.x()) + 160, 100)
        c = QColor(color)
        c.setAlpha(30)
        painter.setBrush(QBrush(c))
        painter.setPen(QPen(c, 1, Qt.PenStyle.DashLine))
        painter.drawRoundedRect(rect, 15, 15)
        painter.setPen(QColor(color))
        painter.setFont(QFont("SansSerif", 8, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(10, 5, 0, 0), Qt.AlignmentFlag.AlignTop, label)

    def draw_node(self, painter, pos, text, color, alpha=255):
        rect = QRectF(pos.x() - 60, pos.y() - 30, 120, 60)
        bg = QColor(color)
        bg.setAlpha(alpha)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawRoundedRect(rect, 10, 10)
        painter.setPen(QColor("#2E3440"))
        painter.setFont(QFont("SansSerif", 8, QFont.Weight.Bold))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def draw_data_tag(self, painter, pos, text):
        painter.setPen(QColor("#ECEFF4"))
        painter.setFont(QFont("SansSerif", 8))
        painter.drawText(pos + QPointF(10, 0), text)

    def draw_arrow(self, painter, p1, p2):
        painter.drawLine(p1, p2)
        angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
        s = 10
        painter.drawLine(p2, p2 - QPointF(s * math.cos(angle - 0.5), s * math.sin(angle - 0.5)))
        painter.drawLine(p2, p2 - QPointF(s * math.cos(angle + 0.5), s * math.sin(angle + 0.5)))

    def lerp(self, p1, p2, t):
        return QPointF(p1.x() * (1-t) + p2.x() * t, p1.y() * (1-t) + p2.y() * t)

class TD3ExpertVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TD3 Explainer: Loss Evaluation")
        self.resize(1100, 850)
        self.setStyleSheet("background-color: #2E3440; color: #ECEFF4;")
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Header
        header = QHBoxLayout()
        header_lbl = QLabel("How Loss is Evaluated")
        header_lbl.setStyleSheet("font-size: 24px; font-weight: bold; color: #88C0D0; margin: 10px;")
        header.addWidget(header_lbl)
        
        self.btn_next = QPushButton("⏭ NEXT PHASE")
        self.btn_next.setMinimumHeight(50)
        self.btn_next.setFixedWidth(200)
        self.btn_next.setStyleSheet("background-color: #88C0D0; color: #2E3440; font-weight: bold; border-radius: 10px;")
        self.btn_next.clicked.connect(self.advance_step)
        header.addWidget(self.btn_next)
        layout.addLayout(header)
        
        # Main Work Area
        content = QHBoxLayout()
        self.canvas = TD3FlowCanvas()
        content.addWidget(self.canvas, stretch=2)
        
        # Math & Logic Panel
        self.math_panel = QFrame()
        self.math_panel.setFixedWidth(380)
        self.math_panel.setStyleSheet("background-color: #3B4252; border-radius: 10px; padding: 20px;")
        self.math_layout = QVBoxLayout(self.math_panel)
        
        self.lbl_phase = QLabel("Phase: IDLE")
        self.lbl_phase.setStyleSheet("font-size: 20px; font-weight: bold; color: #EBCB8B;")
        self.math_layout.addWidget(self.lbl_phase)
        
        self.lbl_formula = QLabel("Equation:")
        self.lbl_formula.setStyleSheet("background-color: #2E3440; color: #A3BE8C; font-family: Monospace; font-size: 13px; padding: 15px; border-radius: 5px;")
        self.math_layout.addWidget(self.lbl_formula)
        
        self.lbl_params = QLabel("Parameter Roles:")
        self.lbl_params.setWordWrap(True)
        self.lbl_params.setStyleSheet("color: #ECEFF4; line-height: 140%; font-size: 11px;")
        self.math_layout.addWidget(self.lbl_params)
        
        self.math_layout.addStretch()
        content.addWidget(self.math_panel)
        layout.addLayout(content)
        
        self.phases = ["COLLECT", "CRITIC_LEARN", "ACTOR_LEARN", "TARGET_SYNC"]
        self.current_idx = 0
        self.steps = 0

    def advance_step(self):
        phase = self.phases[self.current_idx]
        self.steps += 1
        
        self.lbl_phase.setText(f"Phase: {phase}")
        
        if phase == "COLLECT":
            self.lbl_formula.setText("Step: car_angle += action\nReward = -0.1 + Alignment + Distance")
            self.lbl_params.setText(
                "<b>s (Current State):</b> What the car sees now (sensors, target angle).<br>"
                "<b>a (Action):</b> The steering/speed chosen by the Live Actor.<br>"
                "<b>r (Reward):</b> Feedback from the map (Did we get closer? Did we crash?).<br>"
                "<b>s' (Next State):</b> What the car sees AFTER moving."
            )
            self.canvas.set_state(phase)

        elif phase == "CRITIC_LEARN":
            self.lbl_formula.setText("Target_Q = r + γ * min(Target_Q1, Target_Q2)\nLoss = MSE(Live_Q, Target_Q)")
            self.lbl_params.setText(
                "<b>r (Reward):</b> The actual immediate gain.<br>"
                "<b>γ (Gamma):</b> How much we value future rewards (0.99).<br>"
                "<b>s' (Next State):</b> Fed to Target Networks to guess FUTURE value.<br>"
                "<b>min(Q1, Q2):</b> Clipped Double-Q trick to stop overestimation.<br><br>"
                "<i>The Critic learns to predict the total future reward correctly.</i>"
            )
            self.canvas.set_state(phase, show_grads=True)

        elif phase == "ACTOR_LEARN":
            self.lbl_formula.setText("Actor_Loss = -Live_Critic_Q1(s, Live_Actor(s))")
            self.lbl_params.setText(
                "<b>s (State):</b> Sampleled from Replay Buffer.<br>"
                "<b>Live_Actor(s):</b> The current best action the actor can guess.<br>"
                "<b>-Live_Critic_Q1:</b> We use the negative sign because we want to MAXIMIZE the Q-value judged by the critic.<br><br>"
                "<i>The Actor learns to 'please' the Critic.</i>"
            )
            self.canvas.set_state(phase, show_grads=(self.steps % 2 == 0))

        elif phase == "TARGET_SYNC":
            self.lbl_formula.setText("Target = τ * Live + (1 - τ) * Target")
            self.lbl_params.setText(
                "<b>τ (Tau):</b> The soft update rate (0.005).<br><br>"
                "Every cycle, we nudge the Target networks a tiny bit toward the Live ones. This keeps the 'Ground Truth' stable for the next round of learning."
            )
            self.canvas.set_state(phase, show_sync=True)

        self.current_idx = (self.current_idx + 1) % len(self.phases)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TD3ExpertVisualizer()
    window.show()
    sys.exit(app.exec())
