"""
baseline_krillherd.py
======================
"""

import numpy as np


class KrillHerdScheduler:
    name = "KrillHerd"

    def __init__(self, n_krill: int = 15, n_iterations: int = 12, seed: int = None):
        self.n_krill = n_krill
        self.n_iterations = n_iterations
        self.rng = np.random.default_rng(seed)

        
        self.N_max = 0.01         
        self.foraging_speed = 0.02
        self.diffusion_max = 0.005
        self.inertia = 0.6         

    # ------------------------------------------------------------------ #
    def _fitness(self, node_position: float, per_node: np.ndarray, n_actions: int,
                 w_energy: float = 0.4, w_cpu: float = 0.3, w_ram: float = 0.3) -> float:
        idx = int(np.clip(round(node_position), 0, n_actions - 1))
        cpu_avail, ram_avail, energy_used = per_node[idx]
        
        energy_term = energy_used / (energy_used + 1.0)
        return w_energy * energy_term - w_cpu * cpu_avail - w_ram * ram_avail

    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        per_node = state[3:].reshape(-1, 3)[:n_actions]  

        # --- Initialization ---
        positions = self.rng.uniform(0, n_actions, size=self.n_krill)
        most_available_idx = int(np.argmax(per_node[:, 0]))
        positions[0] = most_available_idx

        velocities_induced = np.zeros(self.n_krill)
        velocities_foraging = np.zeros(self.n_krill)

        fitness = np.array([self._fitness(p, per_node, n_actions) for p in positions])
        best_idx = int(np.argmin(fitness))
        best_position = float(positions[best_idx])
        best_fitness = float(fitness[best_idx])

        for _ in range(self.n_iterations):
            
            current_worst = float(fitness.max())
            current_best = float(fitness.min())
            fitness_range = max(current_worst - current_best, 1e-8)

            for i in range(self.n_krill):
                
                induced = 0.0
                for j in range(self.n_krill):
                    if i == j:
                        continue
                    dist = positions[j] - positions[i]
                    if abs(dist) < 1e-8:
                        continue
                    fitness_effect = (fitness[i] - fitness[j]) / fitness_range
                    induced += fitness_effect * np.sign(dist)
                velocities_induced[i] = self.N_max * induced + self.inertia * velocities_induced[i]

                
                foraging_direction = best_position - positions[i]
                velocities_foraging[i] = (self.foraging_speed * np.sign(foraging_direction)
                                           + self.inertia * velocities_foraging[i])

             
                diffusion = self.diffusion_max * self.rng.uniform(-1, 1)

                positions[i] = np.clip(
                    positions[i] + velocities_induced[i] + velocities_foraging[i] + diffusion,
                    0, n_actions - 1e-6,
                )

            fitness = np.array([self._fitness(p, per_node, n_actions) for p in positions])
            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = float(fitness[gen_best_idx])
                best_position = float(positions[gen_best_idx])

        return int(np.clip(round(best_position), 0, n_actions - 1))
