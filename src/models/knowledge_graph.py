# 3. src/models/knowledge_graph.py
from typing import List, Dict, Set
import networkx as nx
import numpy as np

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}
        
    def add_concept(self, concept: str, prerequisites: List[str], embeddings: np.ndarray):
        self.graph.add_node(concept)
        self.concept_embeddings[concept] = embeddings
        
        for prereq in prerequisites:
            self.graph.add_edge(prereq, concept)
    
    def get_prerequisites(self, concept: str) -> Set[str]:
        return set(nx.ancestors(self.graph, concept))
    
    def get_learning_path(self, start_concepts: List[str], target_concept: str) -> List[str]:
        all_paths = []
        for start in start_concepts:
            if nx.has_path(self.graph, start, target_concept):
                paths = list(nx.all_simple_paths(self.graph, start, target_concept))
                all_paths.extend(paths)
        
        if not all_paths:
            return []
        
        # Choose optimal path based on concept similarity and prerequisites
        return min(all_paths, key=len)