import random
import numpy as np
import copy


class GeneticAlgorithm:
    def __init__(
        self,
        food_df,
        monthly_requirements,
        monthly_optimal_ranges,
        budget,
        population_size=100,
        n_genes=None,
        crossover_rate=0.8,
        mutation_rate=0.1,
        n_generations=200,
        tournament_size=5,
        elitism_count=2,
        max_kg_per_food_init=2.0,
        max_kg_per_food_mutation=5.0,
    ):

        self.food_df = food_df
        self.food_names = food_df["Food"].tolist()
        self.nutrient_columns = [
            "Calories_kcal",
            "Protein_g",
            "Fat_g",
            "Carbohydrates_g",
            "Fiber_g",
            "Calcium_mg",
            "Iron_mg",
        ]

        self.monthly_requirements = monthly_requirements
        self.monthly_optimal_ranges = monthly_optimal_ranges
        self.budget = budget

        self.population_size = population_size
        self.n_genes = n_genes if n_genes is not None else len(food_df)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate  # Per-gene mutation rate
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.max_kg_per_food_init = (
            max_kg_per_food_init  # Max kg for a single food in initial population
        )
        self.max_kg_per_food_mutation = (
            max_kg_per_food_mutation  # Max kg for a single food after mutation
        )

        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution = None
        self.best_fitness_overall = -float("inf")

    def _create_individual(self):
        # Creates a random individual (chromosome)
        # Each gene is the quantity (kg) of a food item for the month
        return [
            random.uniform(0, self.max_kg_per_food_init) for _ in range(self.n_genes)
        ]

    def _initialize_population(self):
        self.population = [
            self._create_individual() for _ in range(self.population_size)
        ]

    def _calculate_plan_details(self, individual):
        total_cost = 0
        total_nutrients = {col: 0 for col in self.nutrient_columns}

        for i, quantity_kg in enumerate(individual):
            if quantity_kg > 0:
                food_item = self.food_df.iloc[i]
                total_cost += quantity_kg * food_item["Price_Toman_per_kg"]
                for nutrient in self.nutrient_columns:
                    # food_df nutrients are already per kg
                    total_nutrients[nutrient] += quantity_kg * food_item[nutrient]
        return total_cost, total_nutrients

    def _fitness(self, individual):
        total_cost, achieved_nutrients = self._calculate_plan_details(individual)
        fitness = 10000.0  # Start with a base fitness

        # 1. Budget Constraint (Hard)
        if total_cost > self.budget:
            fitness -= (
                total_cost - self.budget
            ) * 1.0  # Penalize by 1 unit fitness per Toman over budget
            # This is a strong penalty as budget is large

        # 2. Minimum Nutritional Requirements (Hard)
        for nutrient, min_val in self.monthly_requirements.items():
            if achieved_nutrients[nutrient] < min_val:
                # Penalize significantly for not meeting minimums
                fitness -= (
                    min_val - achieved_nutrients[nutrient]
                ) * 20  # Strong penalty factor

        # 3. Optimal Nutrient Ranges (Bonus/Fine-tuning)
        # This part is applied only if minimums are generally met or close to being met.
        # If fitness is already very low due to constraint violations, these bonuses won't matter much.

        # Define weights for how much we care about each nutrient's optimality
        optimal_weights = {
            "Calories_kcal": 1.0,
            "Protein_g": 1.5,
            "Fat_g": 1.0,
            "Carbohydrates_g": 0.8,
            "Fiber_g": 1.2,
            "Calcium_mg": 0.5,
            "Iron_mg": 0.7,
        }

        for nutrient, (opt_low, opt_high) in self.monthly_optimal_ranges.items():
            val = achieved_nutrients[nutrient]
            min_req_val = self.monthly_requirements.get(nutrient, 0)

            if val < min_req_val:  # Already heavily penalized
                continue

            weight = optimal_weights.get(nutrient, 1.0)

            if opt_low <= val <= opt_high:
                fitness += 500 * weight  # Bonus for being in the optimal range
            elif val < opt_low:  # Below optimal but above minimum
                # Penalize for being below optimal but above minimum
                fitness -= (opt_low - val) * 0.5 * weight
            else:  # val > opt_high (Above optimal)
                # Penalize for being too far above optimal, especially for Calories, Fat, Carbs
                if nutrient in ["Calories_kcal", "Fat_g", "Carbohydrates_g"]:
                    fitness -= (val - opt_high) * 0.8 * weight
                else:  # For Protein, Fiber, Calcium, Iron, slightly over optimal might be okay or less penalized
                    fitness -= (val - opt_high) * 0.3 * weight

        # Ensure fitness is not negative, though with high base and large penalties, it can be.
        # This is fine as long as selection handles relative fitness.
        return fitness

    def _selection(self, population_with_fitness):
        # Tournament selection
        selected_parents = []
        for _ in range(
            self.population_size - self.elitism_count
        ):  # Reserve spots for elites
            tournament = random.sample(population_with_fitness, self.tournament_size)
            winner = max(tournament, key=lambda x: x[1])  # x[1] is fitness
            selected_parents.append(winner[0])  # winner[0] is the individual
        return selected_parents

    def _crossover(self, parent1, parent2):
        # Arithmetic/Blend crossover
        if random.random() < self.crossover_rate:
            alpha = random.random()  # Blend factor
            child1 = [
                alpha * p1_gene + (1 - alpha) * p2_gene
                for p1_gene, p2_gene in zip(parent1, parent2)
            ]
            child2 = [
                (1 - alpha) * p1_gene + alpha * p2_gene
                for p1_gene, p2_gene in zip(parent1, parent2)
            ]
            # Ensure non-negativity
            child1 = [max(0, gene) for gene in child1]
            child2 = [max(0, gene) for gene in child2]
            return child1, child2
        return parent1, parent2  # No crossover

    def _mutate(self, individual):
        mutated_individual = list(individual)
        for i in range(self.n_genes):
            if random.random() < self.mutation_rate:
                # Gaussian mutation: add a small random value
                # change = random.gauss(0, 0.5) # mu=0, sigma=0.5 kg
                # mutated_individual[i] += change

                # Or, replace with a new random value within a certain range
                mutated_individual[i] = random.uniform(
                    0, self.max_kg_per_food_mutation * random.random()
                )  # More exploration

                mutated_individual[i] = max(
                    0, mutated_individual[i]
                )  # Ensure non-negative
                mutated_individual[i] = min(
                    mutated_individual[i], self.max_kg_per_food_mutation
                )  # Cap at max
        return mutated_individual

    def run(self):
        self._initialize_population()

        for generation in range(self.n_generations):
            population_with_fitness = []
            current_total_fitness = 0
            for individual in self.population:
                fit = self._fitness(individual)
                population_with_fitness.append((individual, fit))
                current_total_fitness += fit

            population_with_fitness.sort(key=lambda x: x[1], reverse=True)

            current_best_individual, current_best_fitness = population_with_fitness[0]
            self.avg_fitness_history.append(
                current_total_fitness / self.population_size
            )
            self.best_fitness_history.append(current_best_fitness)

            if current_best_fitness > self.best_fitness_overall:
                self.best_fitness_overall = current_best_fitness
                self.best_solution = copy.deepcopy(current_best_individual)

            if (generation + 1) % 10 == 0:
                print(
                    f"Generation {generation+1}/{self.n_generations} | "
                    f"Best Fitness: {current_best_fitness:.2f} | "
                    f"Avg Fitness: {self.avg_fitness_history[-1]:.2f}"
                )

            # Elitism: carry over the best individuals
            next_population = [
                ind[0] for ind in population_with_fitness[: self.elitism_count]
            ]

            # Selection
            parents = self._selection(
                population_with_fitness
            )  # Already accounts for elitism spots

            # Crossover and Mutation
            num_offspring_needed = self.population_size - self.elitism_count
            offspring_count = 0
            parent_idx = 0
            while offspring_count < num_offspring_needed:
                p1 = parents[parent_idx % len(parents)]
                p2 = parents[
                    (parent_idx + 1) % len(parents)
                ]  # Ensure different parent if possible
                parent_idx += 2  # Move to next pair

                child1, child2 = self._crossover(p1, p2)

                next_population.append(self._mutate(child1))
                offspring_count += 1
                if offspring_count < num_offspring_needed:
                    next_population.append(self._mutate(child2))
                    offspring_count += 1

            self.population = next_population[
                : self.population_size
            ]  # Ensure population size is maintained

        print("\nGA Finished.")
        print(f"Best overall fitness: {self.best_fitness_overall:.2f}")
        return self.best_solution, self.best_fitness_history, self.avg_fitness_history
