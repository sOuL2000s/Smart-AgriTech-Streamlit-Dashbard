import numpy as np
from deap import base, creator, tools, algorithms
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt # Added for plotting fitness evolution

# --- Fuzzy Logic Controller (pH, Light → Irrigation Level) ---
# Antecedents (Input variables)
# Expanded range for pH and irrigation to be more robust
ph = ctrl.Antecedent(np.arange(4, 9.1, 0.1), 'pH')
light = ctrl.Antecedent(np.arange(100, 1001, 1), 'light')
# Consequent (Output variable) - Using 0-100 scale for irrigation percentage
irrigation = ctrl.Consequent(np.arange(0, 101, 1), 'irrigation')

# Auto-membership functions for simplicity (can be customized for finer control)
ph.automf(3) # Labels: 'poor', 'average', 'good'
light.automf(3) # Labels: 'poor', 'average', 'good'

# Custom membership functions for irrigation for better clarity and range
irrigation['low'] = fuzz.trimf(irrigation.universe, [0, 0, 30])
irrigation['medium'] = fuzz.trimf(irrigation.universe, [20, 50, 80])
irrigation['high'] = fuzz.trimf(irrigation.universe, [70, 100, 100])

# Fuzzy Rules: Define how inputs map to the output
# Rule 1: If pH is poor OR Light is poor, then irrigation should be high
rule1 = ctrl.Rule(ph['poor'] | light['poor'], irrigation['high'])
# Rule 2: If pH is average AND Light is average, then irrigation should be medium
rule2 = ctrl.Rule(ph['average'] & light['average'], irrigation['medium'])
# Rule 3: If pH is good AND Light is good, then irrigation should be low
rule3 = ctrl.Rule(ph['good'] & light['good'], irrigation['low'])

# Create the control system and simulation
irrigation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
irrigation_sim = ctrl.ControlSystemSimulation(irrigation_ctrl)

def fuzzy_decision(ph_val, light_val):
    """
    Computes the recommended irrigation level based on pH and light intensity
    using the defined fuzzy logic system.
    """
    try:
        irrigation_sim.input['pH'] = ph_val
        irrigation_sim.input['light'] = light_val
        irrigation_sim.compute()
        return irrigation_sim.output['irrigation']
    except ValueError as e:
        # Handle cases where input values are out of defined ranges, or rules don't fire.
        # This can happen if GA explores extreme values. Return a neutral/default.
        # print(f"Fuzzy computation error for pH={ph_val}, light={light_val}: {e}. Returning 50.")
        return 50 # Default to medium irrigation if computation fails

# --- Genetic Algorithm for Optimal Growth Settings ---

# In DEAP, you define your fitness strategy (e.g., maximize or minimize) and individual structure.
# We want to MINIMIZE irrigation level, as lower irrigation (when conditions are good) indicates optimal settings.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # (-1.0,) means minimization
creator.create("Individual", list, fitness=creator.FitnessMin) # Individual linked to FitnessMin

toolbox = base.Toolbox()

# Attribute generators for the individual's genes (pH and Light intensity)
# pH range: 5.5 to 7.5 (common optimal range for many crops)
toolbox.register("attr_ph", np.random.uniform, 5.5, 7.5)
# Light intensity range: 100 to 1000 (example range for crop lighting)
toolbox.register("attr_light", np.random.uniform, 100, 1000)

# Individual creator: a list containing one pH and one light intensity value
def create_individual():
    return [
        toolbox.attr_ph(),    # Index 0: pH
        toolbox.attr_light()  # Index 1: Light intensity
    ]
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
# Population creator: a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation Function: how to assess the "fitness" of an individual
def evaluate(individual):
    """
    Evaluates an individual (pH, light) by using the fuzzy logic controller
    to determine the 'irrigation level' required.
    
    The goal is to find pH and Light conditions that minimize the required irrigation.
    Therefore, the fitness is the negative of the irrigation level.
    """
    ph_val, light_val = individual
    irrigation_level = fuzzy_decision(ph_val, light_val)
    return irrigation_level, # Return as a tuple, DEAP expects this. Since FitnessMin has weight -1.0, this value will be minimized.

toolbox.register("evaluate", evaluate) # Register the evaluation function

# Genetic Operators
toolbox.register("mate", tools.cxBlend, alpha=0.5) # Blended Crossover
toolbox.register("select", tools.selTournament, tournsize=3) # Tournament Selection

# --- Adaptive Mutation Rate Function ---
def adaptive_mutation_rate(current_soil_moisture):
    """
    Calculates an adaptive independent mutation probability (indpb)
    based on the current soil moisture level (representing environmental stress).
    
    - Low soil moisture (<30): High stress, increase mutation rate (0.35) for broader exploration.
    - High soil moisture (>70): Low stress (or potentially over-watering), decrease mutation rate (0.1)
                                to converge on existing good solutions.
    - Moderate soil moisture (30-70): Balanced stress, moderate mutation rate (0.2).
    """
    if current_soil_moisture < 30: # High stress (e.g., drought conditions)
        return 0.35  # Higher mutation to encourage diverse solutions
    elif current_soil_moisture > 70: # Low stress (e.g., potential waterlogging)
        return 0.1   # Lower mutation to fine-tune existing solutions
    else: # Moderate stress (optimal range)
        return 0.2   # Standard mutation rate

# --- CURRENT_SOIL_MOISTURE: This would typically come from your live sensor feed ---
# For demonstration, we'll use a hardcoded value. In a real system, you'd fetch
# the latest soil moisture reading from Firebase (Phase 1).
CURRENT_SOIL_MOISTURE = 45 # Example: Assume current soil moisture is 45%

# Calculate the adaptive independent probability of mutation for the current run
adaptive_indpb = adaptive_mutation_rate(CURRENT_SOIL_MOISTURE)

# Register the mutate operator using the calculated adaptive probability
# mu: mean of the gaussian distribution
# sigma: standard deviation of the gaussian distribution
# indpb: independent probability for each attribute (gene) to be mutated
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=adaptive_indpb)
print(f"Adaptive Mutation Probability (indpb) set to: {adaptive_indpb:.2f} based on current soil moisture {CURRENT_SOIL_MOISTURE}%.")


# --- Run Genetic Algorithm ---
pop_size = 50 # Number of individuals in the population
num_generations = 100 # Number of generations to run the GA

# Initialize the population
pop = toolbox.population(n=pop_size)

# Hall of Fame to store the best individual found during the evolution
hof = tools.HallOfFame(1)

# Statistics to track fitness evolution
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min) # Min fitness (best individual for minimization)
stats.register("max", np.max) # Max fitness (worst individual for minimization)

print("\n--- Running Fuzzy-Adaptive Genetic Algorithm ---")
# The eaSimple algorithm (Evolutionary Algorithm Simple)
# cxpb: Probability of two individuals to crossover.
# mutpb: Probability for an individual to be mutated. (Here set to 1.0 to ensure all individuals get a chance at mutation, then indpb controls gene-level mutation)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=1.0, ngen=num_generations,
                               stats=stats, halloffame=hof, verbose=True)

# --- Best Solution ---
best_ind = hof[0] # The best individual is in the Hall of Fame
print(f"\n✅ Optimization Complete.")
print(f"Optimal pH: {best_ind[0]:.2f}")
print(f"Optimal Light Intensity: {best_ind[1]:.2f} lux")

# Calculate the irrigation level for the best found settings using the fuzzy system
optimal_irrigation_level = fuzzy_decision(best_ind[0], best_ind[1])
print(f"Fuzzy-calculated Irrigation Level for Optimal Settings: {optimal_irrigation_level:.2f}% (Lower is better)")
print(f"Corresponding Fitness Value (Negative Irrigation Level): {best_ind.fitness.values[0]:.2f}")


# --- Optional: Visualize the fitness evolution ---
gen, avg, std, min_fit, max_fit = log.select("gen", "avg", "std", "min", "max")
plt.figure(figsize=(10, 6))
plt.plot(gen, avg, label="Average Fitness")
plt.plot(gen, min_fit, label="Minimum Fitness (Best Individual)") # For minimization, min fitness is best
plt.fill_between(gen, avg - std, avg + std, color='blue', alpha=0.1, label="Std Dev")
plt.xlabel("Generation")
plt.ylabel("Fitness (Irrigation Level - Lower is Better)")
plt.title("Genetic Algorithm Fitness Evolution (Minimizing Irrigation Level)")
plt.legend()
plt.grid(True)
plt.show()