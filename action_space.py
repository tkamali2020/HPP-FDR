"""
action_space.py
================
"""

from dataclasses import dataclass


@dataclass
class ActionSpace:
    n_nodes: int  

    @property
    def n_actions(self) -> int:
        return self.n_nodes + 1  

    @property
    def cloud_action_index(self) -> int:
     
        return self.n_nodes

    def is_cloud_action(self, action: int) -> bool:
        return action == self.cloud_action_index

    def node_index_for_action(self, action: int) -> int:
        
        if not (0 <= action < self.n_nodes):
            raise ValueError(
                f"action {action} is not a fog-node action for N={self.n_nodes} "
                f"(did you forget to check is_cloud_action() first? "
                f"cloud_action_index={self.cloud_action_index})"
            )
        return action

    def valid_actions_mask(self, node_is_alive: list) -> list:
        
        assert len(node_is_alive) == self.n_nodes
        return list(node_is_alive) + [True]
