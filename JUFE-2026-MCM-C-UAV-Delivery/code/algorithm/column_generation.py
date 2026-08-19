import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, LpContinuous

class ColumnGeneration:
    def __init__(self, network, drone_capacity):
        self.network = network
        self.drone_capacity = drone_capacity
        self.N = len(network.nodes)
        self.master_problem = None
        self.columns = []
    
    def initialize_columns(self):
        self.columns = []
        for node in self.network.nodes:
            demand = self.network.demand[node]
            min_flights = int(np.ceil(demand / self.drone_capacity))
            for _ in range(min_flights):
                column = {node: 1}
                for other_node in self.network.nodes:
                    if other_node != node:
                        column[other_node] = 0
                self.columns.append(column)
    
    def solve_master_problem(self):
        self.master_problem = LpProblem("Column_Generation_Master", LpMinimize)
        
        lambda_vars = LpVariable.dicts("lambda", range(len(self.columns)), lowBound=0, cat=LpContinuous)
        
        self.master_problem += lpSum(lambda_vars)
        
        for node in self.network.nodes:
            self.master_problem += lpSum([lambda_vars[i] * self.columns[i][node] for i in range(len(self.columns))]) >= 1
        
        self.master_problem.solve()
        
        duals = {}
        for node in self.network.nodes:
            duals[node] = self.master_problem.constraints[node].pi if hasattr(self.master_problem.constraints[node], 'pi') else 0
        
        return duals, LpStatus[self.master_problem.status]
    
    def solve_pricing_problem(self, duals):
        min_reduced_cost = float('inf')
        best_column = None
        
        for node in self.network.nodes:
            reduced_cost = 1 - duals.get(node, 0)
            if reduced_cost < min_reduced_cost:
                min_reduced_cost = reduced_cost
                best_column = {n: 1 if n == node else 0 for n in self.network.nodes}
        
        return best_column, min_reduced_cost
    
    def solve(self, max_iterations=50):
        self.initialize_columns()
        
        for _ in range(max_iterations):
            duals, status = self.solve_master_problem()
            
            if status != 'Optimal':
                break
            
            new_column, reduced_cost = self.solve_pricing_problem(duals)
            
            if reduced_cost >= 0:
                break
            
            self.columns.append(new_column)
        
        solution = self.extract_solution()
        return solution
    
    def extract_solution(self):
        if self.master_problem is None:
            self.solve_master_problem()
        
        results = {
            'flights': {node: 0 for node in self.network.nodes},
            'total_drones': 0
        }
        
        for i, col in enumerate(self.columns):
            var = self.master_problem.variablesDict().get(f"lambda_{i}")
            if var and var.value() > 0.001:
                for node in self.network.nodes:
                    if col[node] > 0:
                        results['flights'][node] += int(np.round(var.value()))
        
        results['total_drones'] = sum(results['flights'].values())
        return results