import pandas as pd
import json
import matplotlib.pyplot as plt
import os

from simu_annealing import SimulatedAnnealing 

def load_data(food_file_path, needs_file_path):
    """Loads food data and nutritional needs."""
    try:
        food_df = pd.read_csv(food_file_path)

    except FileNotFoundError:
        print(f"Error: Food data file not found at {food_file_path}")
        return None, None
    
    try:
        with open(needs_file_path, 'r') as f:
            nutritional_needs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Nutritional needs file not found at {needs_file_path}")
        return food_df, None
        
    return food_df, nutritional_needs

def main():
    # Construct paths relative to the SA directory
    base_dir = os.path.dirname(__file__)
    food_file = os.path.join(base_dir, '../data/foods.csv')
    needs_file = os.path.join(base_dir, '../data/normal_fit.json')

    # Load data
    food_df, daily_nutritional_needs = load_data(food_file, needs_file)

    if food_df is None or daily_nutritional_needs is None:
        print("Failed to load data. Exiting.")
        return

    # --- Optimal Targets Configuration (Daily Values) ---
    optimal_targets_config = {
        "Calories_target_daily": 1900,  # Slightly less than 2000
        "Calories_weight_over": 0.03,    # Penalty weight if actual > target
        "Calories_ideal_range_kcal_monthly": 30 * 150, # e.g. allow 1850-2000 without penalty if target is 1900

        "Protein_target_daily": 110,    # Slightly more than 100
        "Protein_weight_under": 1.5,    # Penalty weight if actual < target

        "Fat_target_daily": 55,         # Slightly less than 60 (e.g. 55g) | Max is 60g (from daily_nutritional_needs)
        "Fat_weight_over_target": 0.8,  # Penalty if actual > target (but <= 60g)
        "Fat_weight_over_max": 10.0,    # Heavier penalty if actual > 60g

        "Carbs_target_daily": 230,      # Slightly less than 250
        "Carbs_weight_over": 0.04,

        "Fiber_target_daily": 30,       # Slightly more than 25
        "Fiber_weight_under": 2.5,

        "Calcium_target_daily": 1100,   # Slightly more than 1000
        "Calcium_weight_under": 0.003,

        "Iron_target_daily": 20,        # Slightly more than 18
        "Iron_weight_under": 6.0,
    }

    # SA Parameters
    initial_temp = 20000        # Higher initial temp for more exploration
    cooling_rate = 0.99         # Cooling rate
    stopping_temp = 0.01        # When to stop
    max_iter_per_temp = 150     # Iterations at each temperature level

    budget_toman = 4_000_000  # Lily's monthly budget

    # Initialize and run SA
    sa_optimizer = SimulatedAnnealing(
        food_df=food_df,
        nutritional_needs=daily_nutritional_needs,
        budget=budget_toman,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        stopping_temp=stopping_temp,
        max_iter_per_temp=max_iter_per_temp,
        optimal_targets_config=optimal_targets_config
    )

    print("Starting Simulated Annealing optimization...")
    best_solution_quantities, best_solution_energy, history, best_solution_details_log = sa_optimizer.run()
    
    print("\n--- Optimization Finished ---")
    print(f"Best solution energy (lower is better): {best_solution_energy:.2f}")

    # Display the best shopping basket
    print("\nBest Shopping Basket (Monthly Quantities in kg):")
    final_items_cost = 0
    final_selected_items = []
    for i, quantity_kg in enumerate(best_solution_quantities):
        if quantity_kg > 0.001: # Show items with more than 1g
            food_name = food_df.iloc[i].get('Food')
            price_per_kg = food_df.iloc[i].get('Price_Toman_per_kg')
            item_cost = quantity_kg * price_per_kg
            final_items_cost += item_cost
            final_selected_items.append(f"- {food_name}: {quantity_kg:.3f} kg (Cost: {item_cost:,.0f} Toman)")
    
    if not final_selected_items:
        print("No items selected or quantities are too small.")
    else:
        for item_detail in final_selected_items:
            print(item_detail)
    print(f"\nTotal Cost of Best Basket: {final_items_cost:,.0f} Toman (Budget: {budget_toman:,.0f} Toman)")

    # Calculate and display nutrients for the best solution
    best_nutrients_monthly, _ = sa_optimizer.calculate_nutrients_and_cost(best_solution_quantities)
    print("\nNutritional Info for Best Basket (Monthly Totals & Daily Averages):")
    for dn_key, df_col_name in sa_optimizer.needs_mapping.items():
        min_need_monthly = sa_optimizer.monthly_needs[dn_key]
        actual_monthly = best_nutrients_monthly.get(df_col_name, 0)
        actual_daily_avg = actual_monthly / 30
        
        min_need_daily = daily_nutritional_needs[dn_key]
        optimal_target_daily_key = f"{dn_key}_target_daily" # Construct key for optimal_targets_config
        optimal_target_daily_val_str = ""
        if optimal_target_daily_key in optimal_targets_config:
             optimal_target_daily_val_str = f", Optimal Daily Target: ~{optimal_targets_config[optimal_target_daily_key]}"
        
        print(f"- {dn_key} ({df_col_name}):")
        print(f"  Monthly: {actual_monthly:.2f} (Min Required: {min_need_monthly:.2f})")
        print(f"  Daily Avg: {actual_daily_avg:.2f} (Min Required: {min_need_daily:.2f}{optimal_target_daily_val_str})")


    # # --- Plotting ---
    # # 1. Energy vs. Temperature Steps
    # plt.figure(figsize=(18, 12))
    # plt.subplot(2, 2, 1)
    # plt.plot(history['energy'], label='Current Solution Energy')
    # plt.plot(history['best_energy'], label='Best Solution Energy')
    # # plt.plot(history['temperature'], label='Temperature', linestyle=':', color='grey')
    # plt.xlabel('Temperature Cooling Steps (Epochs)')
    # plt.ylabel('Energy (Cost Function Value)')
    # plt.title('SA: Solution Energy Over Temperature Steps')
    # plt.legend()
    # plt.grid(True)

    # # 2. Nutrient values in the best solution over iterations/time
    # # Example: Plotting Protein and Calories from the best_solution_details_log
    # if best_solution_details_log:
    #     iterations = [d['iteration'] for d in best_solution_details_log]
        
    #     # Protein Plot
    #     protein_values_daily = [d['nutrients']['Protein_g']/30 for d in best_solution_details_log]
    #     plt.subplot(2, 2, 2)
    #     plt.plot(iterations, protein_values_daily, marker='.', linestyle='-', markersize=2, label='Daily Avg Protein in Best Solution')
    #     plt.axhline(y=daily_nutritional_needs['Protein'], color='red', linestyle='--', label=f"Min Daily Protein ({daily_nutritional_needs['Protein']}g)")
    #     plt.axhline(y=optimal_targets_config["Protein_target_daily"], color='green', linestyle='--', label=f"Optimal Daily Protein (~{optimal_targets_config['Protein_target_daily']}g)")
    #     plt.xlabel('Iterations (when new best solution found)')
    #     plt.ylabel('Protein (g/day)')
    #     plt.title('SA: Protein in Best Solution Over Iterations')
    #     plt.legend()
    #     plt.grid(True)

    #     # Calories Plot
    #     calories_values_daily = [d['nutrients']['Calories_kcal']/30 for d in best_solution_details_log]
    #     plt.subplot(2, 2, 3)
    #     plt.plot(iterations, calories_values_daily, marker='.', linestyle='-', markersize=2, label='Daily Avg Calories in Best Solution')
    #     plt.axhline(y=daily_nutritional_needs['Calories'], color='red', linestyle='--', label=f"Min Daily Calories ({daily_nutritional_needs['Calories']}kcal)")
    #     plt.axhline(y=optimal_targets_config["Calories_target_daily"], color='green', linestyle='--', label=f"Optimal Daily Calories (~{optimal_targets_config['Calories_target_daily']}kcal)")
    #     plt.xlabel('Iterations (when new best solution found)')
    #     plt.ylabel('Calories (kcal/day)')
    #     plt.title('SA: Calories in Best Solution Over Iterations')
    #     plt.legend()
    #     plt.grid(True)
        
    #     # Cost Plot
    #     cost_values = [d['cost'] for d in best_solution_details_log]
    #     plt.subplot(2, 2, 4)
    #     plt.plot(iterations, cost_values, marker='.', linestyle='-', markersize=2, label='Cost of Best Solution')
    #     plt.axhline(y=budget_toman, color='purple', linestyle='--', label=f"Budget Limit ({budget_toman:,.0f})")
    #     plt.xlabel('Iterations (when new best solution found)')
    #     plt.ylabel('Cost (Toman)')
    #     plt.title('SA: Cost of Best Solution Over Iterations')
    #     plt.legend()
    #     plt.grid(True)


    # plt.tight_layout()
    # plt.savefig(os.path.join(base_dir, 'SA_Optimization_Progress.png'))
    # print(f"\nPlots saved to {os.path.join(base_dir, 'SA_Optimization_Progress.png')}")
    # plt.show()

if __name__ == '__main__':
    main()