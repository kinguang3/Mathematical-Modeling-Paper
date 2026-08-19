import numpy as np
from collections import defaultdict

class SchedulingModel:
    def __init__(self, network, drone_capacity):
        self.network = network
        self.drone_capacity = drone_capacity
        self.schedule = []
        self.drone_status = {}
    
    def generate_initial_schedule(self, num_drones):
        self.drone_status = {f'D{i+1}': {'location': self.network.hub, 'available_time': 0, 'remaining_capacity': self.drone_capacity} 
                             for i in range(num_drones)}
        self.schedule = []
        
        for node in self.network.nodes:
            demand = self.network.demand[node]
            flights_needed = int(np.ceil(demand / self.drone_capacity))
            
            for _ in range(flights_needed):
                drone = self._find_available_drone()
                if drone:
                    self._assign_flight(drone, node)
    
    def _find_available_drone(self):
        available_drones = sorted(self.drone_status.keys(), 
                                key=lambda d: self.drone_status[d]['available_time'])
        return available_drones[0] if available_drones else None
    
    def _assign_flight(self, drone_id, destination):
        drone = self.drone_status[drone_id]
        departure_time = drone['available_time']
        flight_time = self.network.flight_time[(self.network.hub, destination)]
        arrival_time = departure_time + flight_time
        
        flight = {
            'drone_id': drone_id,
            'from': self.network.hub,
            'to': destination,
            'departure': departure_time,
            'arrival': arrival_time,
            'capacity_used': min(self.drone_capacity, self.network.demand[destination])
        }
        
        self.schedule.append(flight)
        
        return_time = self.network.flight_time[(destination, self.network.hub)]
        drone['location'] = self.network.hub
        drone['available_time'] = arrival_time + return_time
        drone['remaining_capacity'] = self.drone_capacity
    
    def reschedule_after_disruption(self, closed_segment, affected_flights):
        new_schedule = [f for f in self.schedule if f not in affected_flights]
        rescheduled_flights = []
        
        for flight in affected_flights:
            alternatives = self._find_alternative_routes(flight['to'])
            if alternatives:
                best_alt = min(alternatives, key=lambda x: x['delay'])
                new_flight = flight.copy()
                new_flight['delay'] = best_alt['delay']
                new_flight['arrival'] += best_alt['delay']
                rescheduled_flights.append(new_flight)
        
        new_schedule.extend(rescheduled_flights)
        self.schedule = new_schedule
        
        return rescheduled_flights
    
    def _find_alternative_routes(self, target_node):
        alternatives = []
        for node in self.network.nodes:
            if node != target_node:
                if (self.network.hub, node) in self.network.capacity:
                    alternatives.append({
                        'via': node,
                        'delay': self.network.flight_time[(self.network.hub, node)] + 
                                self.network.flight_time[(node, target_node)] -
                                self.network.flight_time.get((self.network.hub, target_node), 0)
                    })
        return alternatives
    
    def calculate_total_delay(self):
        return sum(f.get('delay', 0) for f in self.schedule)
    
    def get_schedule_summary(self):
        summary = defaultdict(list)
        for flight in self.schedule:
            summary[flight['drone_id']].append(flight)
        return dict(summary)