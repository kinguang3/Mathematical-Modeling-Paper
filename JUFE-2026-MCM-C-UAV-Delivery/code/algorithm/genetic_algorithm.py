import numpy as np
import random
from collections import defaultdict

class GeneticAlgorithm:
    def __init__(self, network, drone_capacity, pop_size=50, generations=100, mutation_rate=0.1):
        self.network = network
        self.drone_capacity = drone_capacity
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = {}
            for node in self.network.nodes:
                demand = self.network.demand[node]
                individual[node] = int(np.ceil(demand / self.drone_capacity)) + random.randint(-1, 2)
                individual[node] = max(1, individual[node])
            population.append(individual)
        return population
    
    def calculate_fitness(self, individual):
        total_flights = sum(individual.values())
        penalty = 0
        
        for node in self.network.nodes:
            flights = individual[node]
            demand = self.network.demand[node]
            capacity = self.network.capacity[(self.network.hub, node)]
            
            if flights * self.drone_capacity < demand:
                penalty += (demand - flights * self.drone_capacity) * 100
            
            if flights > capacity:
                penalty += (flights - capacity) * 50
        
        return total_flights + penalty
    
    def select_parents(self, population):
        fitness_scores = [(self.calculate_fitness(ind), ind) for ind in population]
        fitness_scores.sort(key=lambda x: x[0])
        parents = [ind for (_, ind) in fitness_scores[:int(self.pop_size * 0.3)]]
        return parents
    
    def crossover(self, parent1, parent2):
        child = {}
        for node in self.network.nodes:
            child[node] = random.choice([parent1[node], parent2[node]])
        return child
    
    def mutate(self, individual):
        for node in self.network.nodes:
            if random.random() < self.mutation_rate:
                individual[node] += random.randint(-1, 1)
                individual[node] = max(1, individual[node])
        return individual
    
    def solve(self):
        population = self.initialize_population()
        
        for gen in range(self.generations):
            parents = self.select_parents(population)
            new_population = []
            
            while len(new_population) < self.pop_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
            current_best = min(population, key=self.calculate_fitness)
            current_fitness = self.calculate_fitness(current_best)
            
            if current_fitness < self.best_fitness:
                self.best_fitness = current_fitness
                self.best_solution = current_best.copy()
        
        return {
            'solution': self.best_solution,
            'fitness': self.best_fitness,
            'total_drones': sum(self.best_solution.values())
        }