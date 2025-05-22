import math
import random
import pandas as pd

class SimulatedAnnealing:
    def __init__(self, food_df, nutritional_needs, budget,
                 initial_temp, cooling_rate, stopping_temp, max_iter_per_temp,
                 optimal_targets_config):
        self.food_df = food_df
        self.num_foods = len(food_df)
        self.monthly_needs = {k: v * 30 for k, v in nutritional_needs.items()} # Convert daily needs to monthly needs
        self.budget = budget
        self.current_temp = initial_temp
        self.initial_temp = initial_temp # Keep a record for adaptive changes
        self.cooling_rate = cooling_rate
        self.stopping_temp = stopping_temp
        self.max_iter_per_temp = max_iter_per_temp

        self.nutrient_columns = ['Calories_kcal', 'Protein_g', 'Fat_g', 'Carbohydrates_g', 'Fiber_g', 'Calcium_mg', 'Iron_mg']
        # Mapping from JSON keys (daily needs) to DataFrame columns
        self.needs_mapping = {
            "Calories": "Calories_kcal", "Protein": "Protein_g", "Fat": "Fat_g",
            "Carbohydrates": "Carbohydrates_g", "Fiber": "Fiber_g",
            "Calcium": "Calcium_mg", "Iron": "Iron_mg"
        }
        self.optimal_targets_config = optimal_targets_config

        # Initialize solution
        self.current_solution = self.generate_initial_solution()
        self.current_energy = self.calculate_energy(self.current_solution)
        self.best_solution = list(self.current_solution)
        self.best_energy = self.current_energy

        self.history = {'temperature': [], 'energy': [], 'best_energy': []}
        self.best_solution_details_over_time = [] # For plotting nutrient evolution
        print(f"Initial monthly needs: {self.monthly_needs}")

    def generate_initial_solution(self):
        """Generates a random list of quantities (in kg) for each food item."""
        # Start with small random quantities (e.g., 0 to 0.5 kg per item). This represents the genotype/state
        solution = [random.uniform(0, 0.5) for _ in range(self.num_foods)]
        return solution

    def calculate_nutrients_and_cost(self, solution_kg):
        """Calculates total nutrients and cost for a given solution (list of food quantities in kg)."""
        total_nutrients = {col: 0 for col in self.nutrient_columns}
        total_cost = 0

        for i, quantity_kg in enumerate(solution_kg):
            if quantity_kg > 0:
                food_item = self.food_df.iloc[i]
                # Nutritional values in food_df are per 100g, quantity is in kg
                # So, quantity_in_100g_units = quantity_kg * 10
                for nutrient_col in self.nutrient_columns:
                    if nutrient_col in food_item and pd.notna(food_item[nutrient_col]):
                        total_nutrients[nutrient_col] += food_item[nutrient_col] * quantity_kg * 10
                
                price_col = 'Price_Toman_per_kg'
                if price_col in food_item and pd.notna(food_item[price_col]):
                     total_cost += food_item[price_col] * quantity_kg
        return total_nutrients, total_cost

    def calculate_energy(self, solution_kg):
        """
        Calculates the energy (cost/fitness) of a solution. Lower energy is better.
        Considers budget, minimum nutritional needs, and optimal targets.
        """
        total_nutrients, total_cost = self.calculate_nutrients_and_cost(solution_kg)
        energy = 0.0

        # Budget Penalty (Tunable weight)
        budget_overspend_penalty_weight = 2.0
        if total_cost > self.budget:
            energy += (total_cost - self.budget) * budget_overspend_penalty_weight

        # Minimum Nutritional Requirements Penalty (Tunable weights)
        min_needs_penalty_weights = {
            "Calories_kcal": 0.05, "Protein_g": 1.0, "Fat_g": 1.0,
            "Carbohydrates_g": 0.05, "Fiber_g": 2.0, "Calcium_mg": 0.002, "Iron_mg": 5.0
        }

        for daily_need_key, monthly_min_val in self.monthly_needs.items():
            df_col_name = self.needs_mapping[daily_need_key]
            current_nutrient_val = total_nutrients.get(df_col_name, 0)
            if current_nutrient_val < monthly_min_val:
                deficit = monthly_min_val - current_nutrient_val
                energy += deficit * min_needs_penalty_weights.get(df_col_name, 1.0) # Penalty based on deficit amount and weight

        # Optimal Nutritional Targets Penalty/Reward
        opt_cfg = self.optimal_targets_config
        
        # Calories: slightly less than 2000/day
        calories_val = total_nutrients.get("Calories_kcal", 0)
        min_calories_monthly = self.monthly_needs["Calories"]
        target_calories_monthly = opt_cfg["Calories_target_daily"] * 30
        if calories_val >= min_calories_monthly:
            if calories_val > target_calories_monthly: 
                energy += (calories_val - target_calories_monthly) * opt_cfg["Calories_weight_over"]
            elif calories_val < target_calories_monthly - (opt_cfg.get("Calories_ideal_range_kcal_monthly", 30*100)): # Penalize if too far under target
                energy += (target_calories_monthly - (opt_cfg.get("Calories_ideal_range_kcal_monthly", 30*100)) - calories_val) * opt_cfg["Calories_weight_over"] * 0.5


        # Protein: slightly more than 100/day
        protein_val = total_nutrients.get("Protein_g", 0)
        min_protein_monthly = self.monthly_needs["Protein"]
        target_protein_monthly = opt_cfg["Protein_target_daily"] * 30
        if protein_val >= min_protein_monthly:
            if protein_val < target_protein_monthly: 
                energy += (target_protein_monthly - protein_val) * opt_cfg["Protein_weight_under"]
        
        # Fat: exactly 60/day or slightly less
        fat_val = total_nutrients.get("Fat_g", 0)
        max_fat_monthly = self.monthly_needs["Fat"] # Fat is 60g*30 which is the max allowed
        target_fat_monthly = opt_cfg["Fat_target_daily"] * 30 
        if fat_val > max_fat_monthly: 
            energy += (fat_val - max_fat_monthly) * opt_cfg["Fat_weight_over_max"]
        elif fat_val > target_fat_monthly: # If over preferred target but under/at max
            energy += (fat_val - target_fat_monthly) * opt_cfg["Fat_weight_over_target"]

        # Carbohydrates: slightly less than 250/day
        carbs_val = total_nutrients.get("Carbohydrates_g", 0)
        min_carbs_monthly = self.monthly_needs["Carbohydrates"]
        target_carbs_monthly = opt_cfg["Carbs_target_daily"] * 30
        if carbs_val >= min_carbs_monthly:
            if carbs_val > target_carbs_monthly:
                energy += (carbs_val - target_carbs_monthly) * opt_cfg["Carbs_weight_over"]

        # Fiber: slightly more than 25/day
        fiber_val = total_nutrients.get("Fiber_g", 0)
        min_fiber_monthly = self.monthly_needs["Fiber"]
        target_fiber_monthly = opt_cfg["Fiber_target_daily"] * 30
        if fiber_val >= min_fiber_monthly:
            if fiber_val < target_fiber_monthly:
                energy += (target_fiber_monthly - fiber_val) * opt_cfg["Fiber_weight_under"]

        # Calcium: slightly more than 1000/day
        calcium_val = total_nutrients.get("Calcium_mg", 0)
        min_calcium_monthly = self.monthly_needs["Calcium"]
        target_calcium_monthly = opt_cfg["Calcium_target_daily"] * 30
        if calcium_val >= min_calcium_monthly:
            if calcium_val < target_calcium_monthly:
                energy += (target_calcium_monthly - calcium_val) * opt_cfg["Calcium_weight_under"]

        # Iron: slightly more than 18/day
        iron_val = total_nutrients.get("Iron_mg", 0)
        min_iron_monthly = self.monthly_needs["Iron"]
        target_iron_monthly = opt_cfg["Iron_target_daily"] * 30
        if iron_val >= min_iron_monthly:
            if iron_val < target_iron_monthly:
                energy += (target_iron_monthly - iron_val) * opt_cfg["Iron_weight_under"]
        
        # Small incentive for lower cost if all other factors are met
        cost_weight = 0.00001 
        energy += total_cost * cost_weight
            
        return energy

    def generate_neighbor_solution(self, current_solution_kg):
        """
        Generates a new solution by making a small change to the current solution.
        """
        neighbor = list(current_solution_kg) # a copy of it!
        
        # Pick one or a few food items to change
        num_changes = random.randint(1, max(1, self.num_foods // 20))
        
        for _ in range(num_changes):
            food_idx = random.randint(0, self.num_foods - 1)
            
            # Adaptive change magnitude based on temperature (more aggressive at high temp)
            # Max change could be e.g. 0.2 kg (200g)
            max_abs_change = 0.2 
            # At high temp, change_factor is near 1, at low temp, near 0 (but not zero)
            change_factor = max(0.1, (self.current_temp / self.initial_temp)**0.5) 
            change = random.uniform(-max_abs_change * change_factor, max_abs_change * change_factor)
            
            neighbor[food_idx] += change
            neighbor[food_idx] = max(0, neighbor[food_idx]) # Ensure quantity is non-negative
            neighbor[food_idx] = round(neighbor[food_idx], 3) # Round to grams

        # Occasionally, try to zero out an item or introduce a small amount of an unused one
        if random.random() < 0.05: # 5% chance
            idx_to_modify = random.randint(0, self.num_foods - 1)
            if neighbor[idx_to_modify] > 0.01 and random.random() < 0.5 : # If it has some quantity, try to zero it
                neighbor[idx_to_modify] = 0.0
            elif neighbor[idx_to_modify] == 0.0 : # If it's zero, try to add a bit
                neighbor[idx_to_modify] = round(random.uniform(0.01, 0.1), 3) # Add 10g to 100g

        return neighbor

    def acceptance_probability(self, old_energy, new_energy):
        """
        Calculates the probability of accepting a new solution.
        """
        if new_energy < old_energy:
            return 1.0
        if self.current_temp <= 1e-6: # Avoid division by zero if temp is effectively zero
            return 0.0
        # Boltzman probability
        return math.exp((old_energy - new_energy) / self.current_temp)

    def run(self):
        """
        Main loop of the SA algorithm.
        """
        num_total_iterations = 0
        while self.current_temp > self.stopping_temp:
            for _ in range(self.max_iter_per_temp):
                num_total_iterations += 1
                neighbor_solution = self.generate_neighbor_solution(self.current_solution)
                neighbor_energy = self.calculate_energy(neighbor_solution)

                # Decide whether to accept the neighbor solution
                if self.acceptance_probability(self.current_energy, neighbor_energy) > random.random():
                    self.current_solution = neighbor_solution
                    self.current_energy = neighbor_energy

                # Update the best solution found so far
                if self.current_energy < self.best_energy:
                    self.best_solution = list(self.current_solution) # a copy of it!
                    self.best_energy = self.current_energy
                    
                    # Log details of new best solution for plotting
                    best_nutrients_now, best_cost_now = self.calculate_nutrients_and_cost(self.best_solution)
                    self.best_solution_details_over_time.append({
                        'iteration': num_total_iterations,
                        'temperature': self.current_temp,
                        'energy': self.best_energy,
                        'cost': best_cost_now,
                        'nutrients': best_nutrients_now
                    })

            # Log history for plotting general progress
            self.history['temperature'].append(self.current_temp)
            self.history['energy'].append(self.current_energy) # Current energy at this temp step
            self.history['best_energy'].append(self.best_energy) # Best energy found up to this temp step

            # Cool down the temperature
            self.current_temp *= self.cooling_rate 
            
            if len(self.history['temperature']) % 50 == 0: # Print progress periodically
                print(f"TempStep: {len(self.history['temperature'])}, Temp: {self.current_temp:.2f}, Energy: {self.current_energy:.2f},\tBest Energy: {self.best_energy:.2f}")
        
        print(f"Finished SA. Total iterations: {num_total_iterations}")
        return self.best_solution, self.best_energy, self.history, self.best_solution_details_over_time