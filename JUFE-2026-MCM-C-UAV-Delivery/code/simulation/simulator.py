import numpy as np
import random
from collections import defaultdict

class DroneSimulator:
    def __init__(self, network, scheduler):
        self.network = network
        self.scheduler = scheduler
        self.drones = {}
        self.event_queue = []
        self.current_time = 0
        self.stats = defaultdict(list)
    
    def initialize_drones(self, num_drones):
        self.drones = {f'D{i+1}': {
            'location': self.network.hub,
            'status': 'idle',
            'current_flight': None,
            'total_deliveries': 0
        } for i in range(num_drones)}
    
    def add_event(self, time, event_type, data):
        self.event_queue.append((time, event_type, data))
        self.event_queue.sort(key=lambda x: x[0])
    
    def process_events(self):
        while self.event_queue and self.event_queue[0][0] <= self.current_time:
            time, event_type, data = self.event_queue.pop(0)
            
            if event_type == 'departure':
                self.handle_departure(data)
            elif event_type == 'arrival':
                self.handle_arrival(data)
            elif event_type == 'disruption':
                self.handle_disruption(data)
    
    def handle_departure(self, data):
        drone_id = data['drone_id']
        target = data['target']
        
        if self.drones[drone_id]['status'] == 'idle':
            self.drones[drone_id]['status'] = 'flying'
            self.drones[drone_id]['current_flight'] = data
            
            flight_time = self.network.flight_time[(self.network.hub, target)]
            arrival_time = self.current_time + flight_time
            self.add_event(arrival_time, 'arrival', {
                'drone_id': drone_id,
                'target': target
            })
    
    def handle_arrival(self, data):
        drone_id = data['drone_id']
        target = data['target']
        
        self.drones[drone_id]['status'] = 'idle'
        self.drones[drone_id]['location'] = target
        self.drones[drone_id]['total_deliveries'] += 1
        self.drones[drone_id]['current_flight'] = None
        
        return_time = self.network.flight_time[(target, self.network.hub)]
        return_time = self.current_time + return_time
        self.add_event(return_time, 'arrival', {
            'drone_id': drone_id,
            'target': self.network.hub
        })
    
    def handle_disruption(self, data):
        closed_segment = data['segment']
        duration = data.get('duration', 60)
        
        for drone_id, drone in self.drones.items():
            if drone['status'] == 'flying':
                flight = drone['current_flight']
                if (flight['from'], flight['target']) == closed_segment:
                    self.drones[drone_id]['status'] = 'delayed'
                    self.add_event(self.current_time + duration, 'resume', {'drone_id': drone_id})
    
    def run(self, duration=180):
        self.current_time = 0
        
        while self.current_time < duration:
            self.process_events()
            
            idle_drones = [d for d in self.drones if self.drones[d]['status'] == 'idle']
            if idle_drones:
                self.scheduler.assign_next_task(idle_drones[0], self.current_time)
            
            self.current_time += 1
            
            for drone_id, drone in self.drones.items():
                self.stats['status'].append({
                    'time': self.current_time,
                    'drone_id': drone_id,
                    'status': drone['status'],
                    'location': drone['location']
                })
        
        return self.get_statistics()
    
    def get_statistics(self):
        stats = {
            'total_deliveries': sum(d['total_deliveries'] for d in self.drones.values()),
            'avg_utilization': self.calculate_utilization(),
            'stats': self.stats
        }
        return stats
    
    def calculate_utilization(self):
        total_time = sum(1 for s in self.stats['status'] if s['status'] == 'flying')
        return total_time / (len(self.drones) * self.current_time)

class WeatherSimulator:
    def __init__(self, network):
        self.network = network
        self.weather_events = []
    
    def generate_weather_events(self, simulation_time):
        events = []
        
        if simulation_time < 120:
            return events
        
        num_events = random.randint(0, max(1, int(simulation_time / 60)))
        
        for _ in range(num_events):
            time = random.randint(30, simulation_time - 30)
            segment = random.choice(list(self.network.flight_time.keys()))
            duration = random.randint(10, min(30, int(simulation_time - time)))
            
            events.append({
                'time': time,
                'type': 'weather',
                'segment': segment,
                'duration': duration
            })
        
        return sorted(events, key=lambda x: x['time'])