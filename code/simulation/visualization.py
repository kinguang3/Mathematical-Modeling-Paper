import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class NetworkVisualizer:
    def __init__(self, network):
        self.network = network
        self.pos = None
    
    def _generate_layout(self):
        if self.pos is None:
            self.pos = nx.spring_layout(self.network.graph)
            self.pos[self.network.hub] = np.array([0.5, 0.9])
            
            for i, node in enumerate(self.network.nodes):
                angle = 2 * np.pi * i / len(self.network.nodes)
                self.pos[node] = np.array([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
        
        return self.pos
    
    def draw_network(self, ax=None, show_labels=True, highlight_edges=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        pos = self._generate_layout()
        
        nx.draw_networkx_nodes(self.network.graph, pos, ax=ax, 
                                node_color='lightblue', 
                                node_size=1500,
                                edgecolors='black')
        
        edge_colors = []
        for edge in self.network.graph.edges():
            if highlight_edges and edge in highlight_edges:
                edge_colors.append('red')
            else:
                edge_colors.append('gray')
        
        nx.draw_networkx_edges(self.network.graph, pos, ax=ax,
                                edge_color=edge_colors,
                                width=2,
                                arrows=True)
        
        if show_labels:
            labels = {node: node for node in self.network.graph.nodes()}
            nx.draw_networkx_labels(self.network.graph, pos, ax=ax,
                                    font_size=12, font_weight='bold')
            
            edge_labels = {(u, v): f"{self.network.flight_time[(u, v)]}min" 
                          for u, v in self.network.graph.edges()}
            nx.draw_networkx_edge_labels(self.network.graph, pos, ax=ax,
                                        edge_labels=edge_labels,
                                        font_color='darkred')
        
        ax.set_title('Drone Delivery Network')
        ax.axis('off')
        
        return ax.figure
    
    def plot_demand_distribution(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        
        nodes = self.network.nodes
        demands = [self.network.demand[node] for node in nodes]
        
        ax.bar(nodes, demands, color='skyblue', edgecolor='black')
        ax.set_xlabel('起降点')
        ax.set_ylabel('需求量 (件/小时)')
        ax.set_title('各起降点需求分布')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        return ax.figure

class ScheduleVisualizer:
    def __init__(self, schedule):
        self.schedule = schedule
    
    def plot_gantt_chart(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        drones = sorted(set(f['drone_id'] for f in self.schedule))
        drone_y = {drone: i for i, drone in enumerate(drones)}
        
        for flight in self.schedule:
            start = flight['departure']
            duration = flight['arrival'] - start
            y_pos = drone_y[flight['drone_id']]
            
            ax.barh(y_pos, duration, left=start, height=0.6,
                    label=f"{flight['from']}→{flight['to']}",
                    alpha=0.8)
        
        ax.set_yticks(range(len(drones)))
        ax.set_yticklabels(drones)
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('无人机')
        ax.set_title('无人机调度甘特图')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        return ax.figure

class ResilienceVisualizer:
    def __init__(self, metrics):
        self.metrics = metrics
    
    def plot_resilience_curve(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        nodes = list(self.metrics.keys())
        phi_values = [self.metrics[node] for node in nodes]
        
        ax.bar(nodes, phi_values, color='salmon', edgecolor='black')
        ax.axhline(y=0.8, color='green', linestyle='--', label='韧性阈值')
        ax.set_xlabel('失效起降点')
        ax.set_ylabel('韧性指标 Φ')
        ax.set_title('各起降点失效后的网络韧性')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        return ax.figure