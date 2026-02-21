import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np

# Add local path to sys.path so we can import pdworld
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_dir)

from pdworld.core.qtable import QTable
from pdworld.core.world import reset_world, apply_action, applicable_actions
from pdworld.core.policies import choose_action
from pdworld.core.learners import q_learning_update, sarsa_update
from pdworld.core.state_mapping import world_state_to_id, state_to_id
from pdworld.core.types import Policy, LearnerType, Action
from pdworld.core.constants import GRID_SIZE, PICKUP_LOCATIONS, DROPOFF_LOCATIONS, ACTIONS

PORT = 8080

class AppState:
    def __init__(self):
        self.rng = np.random.default_rng(42)
        self.q_table = QTable()
        self.world_state = reset_world()
        self.policy = Policy.PRANDOM
        self.learner = LearnerType.Q_LEARNING
        self.alpha = 0.3
        self.gamma = 0.5
        self.step_count = 0
        self.bank = 0.0
        self.terminal_hits = 0

state = AppState()

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            index_path = os.path.join(current_dir, 'index.html')
            with open(index_path, 'rb') as f:
                self.wfile.write(f.read())
        elif self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            data = self.get_state_json()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body) if body else {}

        if self.path == '/api/step':
            num_steps = data.get('num_steps', 1)
            for _ in range(num_steps):
                self.perform_step()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.get_state_json()).encode('utf-8'))
        
        elif self.path == '/api/config':
            if 'policy' in data:
                state.policy = Policy(data['policy'])
            if 'learner' in data:
                state.learner = LearnerType(data['learner'])
            if 'alpha' in data:
                state.alpha = float(data['alpha'])
            if 'gamma' in data:
                state.gamma = float(data['gamma'])
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.get_state_json()).encode('utf-8'))

        elif self.path == '/api/reset':
            hard = data.get('hard', False)
            if hard:
                state.q_table = QTable()
                state.step_count = 0
                state.bank = 0.0
                state.terminal_hits = 0
            state.world_state = reset_world()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.get_state_json()).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def perform_step(self):
        state_id = world_state_to_id(state.world_state)
        action = choose_action(state.policy, state.world_state, state_id, state.q_table, state.rng)
        
        next_world, reward, terminal = apply_action(state.world_state, action)
        next_state_id = world_state_to_id(next_world)

        if state.learner == LearnerType.Q_LEARNING:
            q_learning_update(
                q_table=state.q_table,
                state_id=state_id,
                action=action,
                reward=reward,
                next_state_id=next_state_id,
                next_applicable_actions=applicable_actions(next_world),
                alpha=state.alpha,
                gamma=state.gamma,
                terminal=terminal,
            )
        else: # SARSA
            if terminal:
                next_action = None
            else:
                next_action = choose_action(state.policy, next_world, next_state_id, state.q_table, state.rng)
            
            sarsa_update(
                q_table=state.q_table,
                state_id=state_id,
                action=action,
                reward=reward,
                next_state_id=next_state_id,
                next_action=next_action,
                alpha=state.alpha,
                gamma=state.gamma,
                terminal=terminal,
            )

        state.bank += reward
        state.step_count += 1
        
        if terminal:
            state.terminal_hits += 1
            state.world_state = reset_world()
        else:
            state.world_state = next_world

    def get_state_json(self):
        # We also want to send the Q-values for the current carrying mode
        carrying = state.world_state.carrying
        q_values_grid = []
        for r in range(1, GRID_SIZE + 1):
            row_q = []
            for c in range(1, GRID_SIZE + 1):
                x = 1 if carrying else 0
                if x == 0:
                    s = int(state.world_state.pickup_counts[0] > 0)
                    t = int(state.world_state.pickup_counts[1] > 0)
                    u = int(state.world_state.pickup_counts[2] > 0)
                else:
                    s = int(state.world_state.dropoff_counts[0] < 5)
                    t = int(state.world_state.dropoff_counts[1] < 5)
                    u = int(state.world_state.dropoff_counts[2] < 5)
                s_id = state_to_id(r, c, x, s, t, u)
                
                cell_q = {}
                for a in ACTIONS:
                    cell_q[a.value] = state.q_table.get(s_id, a)
                row_q.append(cell_q)
            q_values_grid.append(row_q)

        return {
            "world": {
                "row": state.world_state.row,
                "col": state.world_state.col,
                "carrying": state.world_state.carrying,
                "pickup_counts": state.world_state.pickup_counts,
                "dropoff_counts": state.world_state.dropoff_counts,
            },
            "stats": {
                "step": state.step_count,
                "bank": state.bank,
                "terminals": state.terminal_hits,
            },
            "config": {
                "policy": state.policy.value,
                "learner": state.learner.value,
                "alpha": state.alpha,
                "gamma": state.gamma,
            },
            "grid_q": q_values_grid,
            "pickups": PICKUP_LOCATIONS,
            "dropoffs": DROPOFF_LOCATIONS,
        }

if __name__ == '__main__':
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Serving PD-World Visualization at http://localhost:{PORT}")
    httpd.serve_forever()
