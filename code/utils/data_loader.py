import csv
import json
import pickle
import numpy as np

class DataLoader:
    @staticmethod
    def load_city_network(filepath):
        nodes = []
        edges = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                edges.append({
                    'from': row['from'],
                    'to': row['to'],
                    'time': float(row['time']),
                    'capacity': int(row['capacity'])
                })
                if row['from'] not in nodes:
                    nodes.append(row['from'])
                if row['to'] not in nodes:
                    nodes.append(row['to'])
        
        return nodes, edges
    
    @staticmethod
    def load_demand_pattern(filepath):
        demand = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                demand[row['node']] = float(row['demand'])
        
        return demand
    
    @staticmethod
    def load_drone_specs(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            specs = json.load(f)
        
        return specs
    
    @staticmethod
    def save_network_graph(graph, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
    
    @staticmethod
    def load_network_graph(filepath):
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
        return graph
    
    @staticmethod
    def generate_sample_data(N=5):
        flight_time = np.zeros((N+1, N+1))
        capacity = np.zeros((N+1, N+1))
        demand = np.zeros(N)
        
        np.random.seed(42)
        
        for i in range(N+1):
            for j in range(N+1):
                if i != j:
                    flight_time[i][j] = np.random.randint(5, 20)
                    capacity[i][j] = np.random.randint(5, 15)
        
        for i in range(N):
            demand[i] = np.random.randint(50, 200)
        
        return flight_time, capacity, demand
    
    @staticmethod
    def save_results(results, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
    @staticmethod
    def load_results(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results