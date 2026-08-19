import numpy as np
from scipy.optimize import linprog
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus

class CapacityPlanningModel:
    def __init__(self, network, drone_capacity):
        self.network = network
        self.drone_capacity = drone_capacity
        self.N = len(network.nodes)
        self.model = None
        self.variables = {}
    
    def build_lp_model(self):
        self.model = LpProblem("Minimize_Drone_Fleet", LpMinimize)
        
        x = LpVariable.dicts("x", self.network.nodes, lowBound=0, cat='Integer')
        y = LpVariable("y", lowBound=0, cat='Integer')
        
        self.model += lpSum([x[node] for node in self.network.nodes]) == y
        
        for node in self.network.nodes:
            self.model += x[node] * self.drone_capacity >= self.network.demand[node]
            self.model += x[node] <= self.network.capacity[(self.network.hub, node)]
        
        self.variables['x'] = x
        self.variables['y'] = y
    
    def solve(self):
        if self.model is None:
            self.build_lp_model()
        
        self.model.solve()
        
        results = {
            'status': LpStatus[self.model.status],
            'total_drones': int(self.variables['y'].value()),
            'flights': {node: int(self.variables['x'][node].value()) for node in self.network.nodes}
        }
        return results
    
    def calculate_min_drones_analytical(self):
        total_flights = 0
        for node in self.network.nodes:
            demand = self.network.demand[node]
            capacity = self.network.capacity[(self.network.hub, node)]
            flights_needed = np.ceil(demand / self.drone_capacity)
            flights_allowed = capacity
            total_flights += min(flights_needed, flights_allowed)
        
        return int(total_flights)
    
    def get_scheduling_plan(self):
        results = self.solve()
        plan = []
        
        for node in self.network.nodes:
            flights = results['flights'][node]
            for i in range(flights):
                plan.append({
                    'flight_id': f"F{node}{i+1}",
                    'from': self.network.hub,
                    'to': node,
                    'capacity_used': min(self.drone_capacity, self.network.demand[node] / flights if flights > 0 else 0)
                })
        
        return plan, results['total_drones']