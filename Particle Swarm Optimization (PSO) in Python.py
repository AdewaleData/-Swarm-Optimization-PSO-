import numpy as np

# Define the problem function
def problem_func(x):
    return (x[0]**2 + x[1]**2)

# Define the PSO parameters
num_particles = 20
max_iterations = 50
c1 = 2.0
c2 = 2.0
w = 0.7
bounds = [(-10, 10), (-10, 10)]

# Define the Particle class
class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.array([np.random.uniform(-1, 1) for _ in range(len(bounds))])
        self.best_position = self.position
        self.best_fitness = float('inf')
        self.fitness = float('inf')

    def update_position(self, bounds):
        self.position = np.clip(self.position + self.velocity, bounds[0], bounds[1])

    def update_velocity(self, global_best_position):
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        self.velocity = w * self.velocity + c1 * r1 * (self.best_position - self.position) + c2 * r2 * (global_best_position - self.position)

    def evaluate_fitness(self, problem_func):
        self.fitness = problem_func(self.position)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position

# Define the PSO function
def pso(problem_func, num_particles, max_iterations, c1, c2, w, bounds):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = None
    global_best_fitness = float('inf')
    for i in range(max_iterations):
        for particle in particles:
            particle.evaluate_fitness(problem_func)
            if particle.fitness < global_best_fitness:
                global_best_fitness = particle.fitness
                global_best_position = particle.position
            particle.update_velocity(global_best_position)
            particle.update_position(bounds)
    return global_best_position, global_best_fitness

# Run the PSO algorithm
best_position, best_fitness = pso(problem_func, num_particles, max_iterations, c1, c2, w, bounds)

print('Best position:', best_position)
print('Best fitness:', best_fitness)
