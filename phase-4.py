# Phase 4: Optimization Engine (Genetic Algorithm + Fuzzy Logic)

import numpy as np
from deap import base, creator, tools, algorithms
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# --- Fuzzy Logic Controller (pH, Light → Irrigation Level) ---
ph = ctrl.Antecedent(np.arange(4, 9, 0.1), 'pH')
light = ctrl.Antecedent(np.arange(100, 1001, 1), 'light')
irrigation = ctrl.Consequent(np.arange(0, 11, 1), 'irrigation')

ph.automf(3)
light.automf(3)
irrigation.automf(3)

rule1 = ctrl.Rule(ph['poor'] | light['poor'], irrigation['high'])
rule2 = ctrl.Rule(ph['average'] & light['average'], irrigation['medium'])
rule3 = ctrl.Rule(ph['good'] & light['good'], irrigation['low'])

irrigation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
irrigation_sim = ctrl.ControlSystemSimulation(irrigation_ctrl)

def fuzzy_decision(ph_val, light_val):
    irrigation_sim.input['pH'] = ph_val
    irrigation_sim.input['light'] = light_val
    irrigation_sim.compute()
    return irrigation_sim.output['irrigation']

# --- Genetic Algorithm for Optimal Growth Settings ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 5.5, 7.5)   # pH range

def create_individual():
    return [
        np.random.uniform(5.5, 7.5),    # pH
        np.random.uniform(100, 1000)   # Light intensity
    ]

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    ph_val, light_val = individual
    irrigation_score = fuzzy_decision(ph_val, light_val)
    return irrigation_score,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Run Genetic Algorithm ---
pop = toolbox.population(n=20)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

# --- Best Solution ---
best_ind = tools.selBest(pop, 1)[0]
print(f"✅ Optimal pH & Light: {best_ind}")
