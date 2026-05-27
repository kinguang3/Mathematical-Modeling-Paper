import numpy as np
from collections import deque

class RollingHorizonScheduler:
    def __init__(self, network, drone_capacity, horizon=3):
        self.network = network
        self.drone_capacity = drone_capacity
        self.horizon = horizon
        self.current_time = 0
        self.drone_status = {}
    
    def initialize_drones(self, num_drones):
        self.drone_status = {f'D{i+1}': {
            'location': self.network.hub,
            'available_time': 0,
            'remaining_capacity': self.drone_capacity
        } for i in range(num_drones)}
    
    def predict_demand(self, time):
        base_demand = {node: self.network.demand[node] for node in self.network.nodes}
        return base_demand
    
    def solve_one_horizon(self, horizon_demand):
        schedule = []
        
        for node in self.network.nodes:
            demand = horizon_demand[node]
            flights_needed = int(np.ceil(demand / self.drone_capacity))
            
            for _ in range(flights_needed):
                drone = self._find_best_drone(node)
                if drone:
                    schedule.append(self._create_flight(drone, node))
        
        return schedule
    
    def _find_best_drone(self, target_node):
        best_drone = None
        best_score = float('inf')
        
        for drone_id, status in self.drone_status.items():
            if status['available_time'] <= self.current_time:
                flight_time = self.network.flight_time.get((status['location'], target_node), float('inf'))
                score = flight_time
                if score < best_score:
                    best_score = score
                    best_drone = drone_id
        
        return best_drone
    
    def _create_flight(self, drone_id, target_node):
        drone = self.drone_status[drone_id]
        departure_time = max(self.current_time, drone['available_time'])
        flight_time = self.network.flight_time[(drone['location'], target_node)]
        arrival_time = departure_time + flight_time
        
        flight = {
            'drone_id': drone_id,
            'from': drone['location'],
            'to': target_node,
            'departure': departure_time,
            'arrival': arrival_time,
            'capacity_used': self.drone_capacity
        }
        
        return_time = self.network.flight_time.get((target_node, self.network.hub), flight_time)
        drone['location'] = self.network.hub
        drone['available_time'] = arrival_time + return_time
        
        return flight
    
    def handle_disruption(self, closed_segment, affected_drones):
        for drone_id in affected_drones:
            self.drone_status[drone_id]['available_time'] += 60
    
    def run(self, num_drones, total_time=24):
        self.initialize_drones(num_drones)
        all_schedules = []
        
        while self.current_time < total_time * 60:
            horizon_demand = self.predict_demand(self.current_time)
            schedule = self.solve_one_horizon(horizon_demand)
            all_schedules.extend(schedule)
            
            if schedule:
                min_arrival = min(f['arrival'] for f in schedule)
                self.current_time = min_arrival
            else:
                self.current_time += 60
        
        return all_schedules