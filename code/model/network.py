import numpy as np
import networkx as nx

class DeliveryNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.hub = 'H'
        self.nodes = []
        self.flight_time = {}
        self.capacity = {}
        self.demand = {}
    
    def add_node(self, node_id, demand=0):
        if node_id != self.hub:
            self.nodes.append(node_id)
        self.graph.add_node(node_id)
        self.demand[node_id] = demand
    
    def add_edge(self, from_node, to_node, time, capacity):
        self.graph.add_edge(from_node, to_node, weight=time, capacity=capacity)
        self.flight_time[(from_node, to_node)] = time
        self.capacity[(from_node, to_node)] = capacity
    
    def build_from_matrix(self, N, flight_time_matrix, capacity_matrix, demand_vector):
        self.add_node(self.hub, demand=0)
        for i in range(N):
            node_id = f'P{i+1}'
            self.add_node(node_id, demand=demand_vector[i])
            self.add_edge(self.hub, node_id, flight_time_matrix[0][i+1], capacity_matrix[0][i+1])
            self.add_edge(node_id, self.hub, flight_time_matrix[i+1][0], capacity_matrix[i+1][0])
    
    def get_shortest_path(self, source, target):
        try:
            path = nx.shortest_path(self.graph, source, target, weight='weight')
            length = nx.shortest_path_length(self.graph, source, target, weight='weight')
            return path, length
        except nx.NetworkXNoPath:
            return None, float('inf')
    
    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))
    
    def get_total_demand(self):
        return sum(self.demand.values())
    
    def __repr__(self):
        return f"DeliveryNetwork(hub={self.hub}, nodes={self.nodes}, edges={len(self.graph.edges)})"