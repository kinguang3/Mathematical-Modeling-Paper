import numpy as np
from collections import defaultdict

class MetricsCalculator:
    @staticmethod
    def calculate_resilience(network, failed_node):
        original_demand = sum(network.demand.values())
        
        if failed_node == network.hub:
            return 0.0
        
        remaining_demand = sum(network.demand[node] for node in network.nodes if node != failed_node)
        
        return remaining_demand / original_demand if original_demand > 0 else 0.0
    
    @staticmethod
    def calculate_network_resilience(network):
        resilience = {}
        
        for node in network.nodes:
            phi = MetricsCalculator.calculate_resilience(network, node)
            resilience[node] = phi
        
        return resilience
    
    @staticmethod
    def identify_critical_nodes(network, top_k=3):
        resilience = MetricsCalculator.calculate_network_resilience(network)
        sorted_nodes = sorted(resilience.items(), key=lambda x: x[1])
        return sorted_nodes[:top_k]
    
    @staticmethod
    def calculate_delay(schedule):
        total_delay = sum(f.get('delay', 0) for f in schedule)
        avg_delay = total_delay / len(schedule) if schedule else 0
        max_delay = max(f.get('delay', 0) for f in schedule) if schedule else 0
        
        return {
            'total_delay': total_delay,
            'average_delay': avg_delay,
            'max_delay': max_delay
        }
    
    @staticmethod
    def calculate_utilization(schedule, total_time):
        if not schedule:
            return 0.0
        
        active_time = sum(f['arrival'] - f['departure'] for f in schedule)
        num_drones = len(set(f['drone_id'] for f in schedule))
        
        return active_time / (num_drones * total_time)
    
    @staticmethod
    def calculate_throughput(schedule, duration):
        deliveries = len(schedule)
        return deliveries / duration if duration > 0 else 0
    
    @staticmethod
    def evaluate_schedule(schedule, total_time):
        metrics = {
            'delay': MetricsCalculator.calculate_delay(schedule),
            'utilization': MetricsCalculator.calculate_utilization(schedule, total_time),
            'throughput': MetricsCalculator.calculate_throughput(schedule, total_time),
            'total_flights': len(schedule),
            'num_drones': len(set(f['drone_id'] for f in schedule))
        }
        
        return metrics

class RobustnessAnalyzer:
    @staticmethod
    def analyze_single_node_failure(network):
        results = {}
        
        for node in network.nodes:
            resilience = MetricsCalculator.calculate_resilience(network, node)
            results[node] = {
                'resilience': resilience,
                'demand_lost': network.demand[node],
                'impact': 'high' if resilience < 0.6 else 'medium' if resilience < 0.8 else 'low'
            }
        
        return results
    
    @staticmethod
    def analyze_multiple_failures(network, failure_nodes):
        original_demand = sum(network.demand.values())
        remaining_demand = sum(network.demand[node] for node in network.nodes if node not in failure_nodes)
        
        return remaining_demand / original_demand if original_demand > 0 else 0.0
    
    @staticmethod
    def evaluate_reinforcement_strategy(network, reinforcement_nodes):
        base_resilience = MetricsCalculator.calculate_network_resilience(network)
        
        improved_network = type(network)()
        for node in network.nodes:
            improved_network.add_node(node, network.demand[node])
        
        for edge in network.graph.edges():
            improved_network.add_edge(edge[0], edge[1],
                                    network.flight_time[edge],
                                    network.capacity[edge] * 2 if edge[1] in reinforcement_nodes else network.capacity[edge])
        
        improved_resilience = MetricsCalculator.calculate_network_resilience(improved_network)
        
        improvement = {}
        for node in base_resilience:
            improvement[node] = improved_resilience.get(node, 0) - base_resilience[node]
        
        return {
            'base': base_resilience,
            'improved': improved_resilience,
            'improvement': improvement
        }