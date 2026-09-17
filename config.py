"""
config.py
=========
 ALL hyperparameters used across HPP-FDRL.
"""

from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------- #
# 1) FUZZY (Takagi-Sugeno) SYSTEM CONFIG
# --------------------------------------------------------------------------- #
@dataclass
class FuzzyConfig:
    
    input_names: List[str] = field(default_factory=lambda: [
        "age", "thallium_stress_test", "blood_pressure", "cholesterol", "chest_pain"
    ])

    
    n_mfs_per_input: List[int] = field(default_factory=lambda: [3, 3, 3, 3, 4])

 
    mf_type: str = "triangular"

    # Total rule count = 3*3*3*3*4 = 324 
    n_rules: int = 324

    
    discretization_bounds: List[float] = field(default_factory=lambda: [1 / 3, 2 / 3])
    priority_labels: List[str] = field(default_factory=lambda: ["Low", "Medium", "High"])

    
    regression_method: str = "ordinary_least_squares"

   
    missing_value_strategy: str = "mean_imputation"   
    normalization: str = "min_max"                    
    train_test_split: float = 0.8                     


# --------------------------------------------------------------------------- #
# 2) ENVIRONMENT / SIMULATION CONFIG
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    
    n_fog_nodes_static: int = 20          
    n_fog_nodes_dynamic_min: int = 20    
    n_fog_nodes_dynamic_max: int = 100
    n_cloud_nodes: int = 1

    workload_min_cases: int = 30         
    workload_max_cases: int = 100        

    request_size_min_kb: int = 20         
    request_size_max_kb: int = 150        

   
    propagation_delay_seconds: float = 0.002

 
    patient_interarrival_seconds: float = 4.0

    
    cpu_profile: str = "Intel i7"
    ram_gb: int = 8
    bandwidth_mbps: int = 100

    scenario: str = "static"              

    node_cpu_total_range: tuple = (4, 16)      
    node_ram_total_range: tuple = (8, 64)      
    node_energy_total_range: tuple = (500, 2000)  #

    random_seed: int = 30                


# --------------------------------------------------------------------------- #
# 2b) CLOUD CONFIG 
# --------------------------------------------------------------------------- #
@dataclass
class CloudConfig:
    
    cpu_total: float = 100000.0     
    ram_total: float = 100000.0     
    energy_total: float = 1e9       

    
    propagation_delay_seconds: float = 0.05

    
    bandwidth_mbps: float = 50.0

    p_idle_watts: float = 5.0
    p_max_watts: float = 80.0

    always_alive: bool = True


# --------------------------------------------------------------------------- #
# 3b) ENERGY MODEL CONFIG 
# --------------------------------------------------------------------------- #
@dataclass
class EnergyConfig:
   
    p_idle_watts: float = 10.0
    p_max_watts: float = 150.0

    
    p_tx_watts: float = 0.5

    bits_per_kb: float = 8000.0  


# --------------------------------------------------------------------------- #
# 3) DEADLINE MODEL CONFIG  
# --------------------------------------------------------------------------- #
@dataclass
class DeadlineConfig:
 
    d_max_seconds: float = 300.0   
    d_min_seconds: float = 30.0    

    def deadline_for_priority(self, p_t: float) -> float:
        """D_i = D_max - (D_max - D_min) * P_t   (Eq. proposed, see docstring)"""
        p_t = min(max(p_t, 0.0), 1.0)
        return self.d_max_seconds - (self.d_max_seconds - self.d_min_seconds) * p_t


# --------------------------------------------------------------------------- #
# 4) REWARD CONFIG  
# --------------------------------------------------------------------------- #
@dataclass
class RewardConfig:
    
   
    w1_cpu: float = 0.20       
    w2_ram: float = 0.18       
    w3_energy: float = 0.22    
    w4_deadline: float = 0.25  
    w5_queue: float = 0.15     


    queue_norm_seconds: float = 60.0

    def validate(self):
        assert self.w1_cpu >= 0 and self.w2_ram >= 0
        assert self.w3_energy >= 0 and self.w4_deadline >= 0
        assert self.w5_queue >= 0
       

# --------------------------------------------------------------------------- #
# 5) ACTOR-CRITIC NETWORK + TRAINING CONFIG
# --------------------------------------------------------------------------- #
@dataclass
class ActorCriticConfig:
 
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    hidden_activation: str = "relu"       
    actor_output_activation: str = "softmax"  
    critic_output_activation: str = "linear"  
    optimizer: str = "adam"              
    
   
    discount_gamma: float = 0.99          
    actor_lr: float = 2e-4               
    critic_lr: float = 1e-3              
    entropy_coef: float = 0.01            
    grad_clip_norm: float = 0.5           
    weight_init: str = "xavier_uniform"   

    # ---- advantage / update rule  ----
    advantage_method: str = "td_error"    
    update_style: str = "online_td0"      
    trajectory_length: int = 1           
    # ---- action selection  ----
    train_action_selection: str = "sample_from_softmax"  
    test_action_selection: str = "greedy"                

    # ---- training length  ----
    n_episodes: int = 100

    convergence_window: int = 10          
    convergence_rel_tolerance: float = 0.01  
    convergence_patience: int = 5         

# --------------------------------------------------------------------------- #
# 6) EVALUATION / EXPERIMENT CONFIG 
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    n_seeds_for_stats: int = 30            
    significance_alpha: float = 0.05     
    scalability_node_counts: List[int] = field(
        default_factory=lambda: [20, 40, 60, 80, 100])  

   
    failure_prob: float = 0.05
    ablation_conditions: List[str] = field(default_factory=lambda: [
        "full_model",
        "no_fuzzy_priority",     
        "no_deadline_term",      
        "no_queue_term",         
        "fixed_reward_weights",  
        "random_policy",         
    ])


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
@dataclass
class HPPFDRLConfig:
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    deadline: DeadlineConfig = field(default_factory=DeadlineConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


CONFIG = HPPFDRLConfig()
