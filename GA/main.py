import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from genetic_algo import GeneticAlgorithm


def load_and_preprocess_food_data(csv_path="data/foods.csv"):
    df = pd.read_csv(csv_path)
    # Nutrient values are per 100g, price is per kg.
    # Convert nutrient values to be per kg for easier calculation with quantities in kg.
    nutrient_cols = [
        "Calories_kcal",
        "Protein_g",
        "Fat_g",
        "Carbohydrates_g",
        "Fiber_g",
        "Calcium_mg",
        "Iron_mg",
    ]
    for col in nutrient_cols:
        df[col] = df[col] * 10  # 100g -> 1000g (1kg)
    return df


def define_requirements():
    # Daily minimum requirements
    daily_min_req = {
        "Calories_kcal": 2000,
        "Protein_g": 100,
        "Fat_g": 60,
        "Carbohydrates_g": 250,
        "Fiber_g": 25,
        "Calcium_mg": 1000,
        "Iron_mg": 18,
    }
    # Monthly minimum requirements (30 days)
    monthly_min_req = {k: v * 30 for k, v in daily_min_req.items()}

    # Daily optimal ranges (lower_bound, upper_bound)
    # These are goals, not hard constraints like minimums.
    # Calories: کمی کمتر از 2000 (e.g., 1800-1999)
    # Protein: کمی بیشتر از 100 (e.g., 101-120)
    # Fat: دقیقاً 60 گرم یا کمی کمتر (e.g., 50-60)
    # Carbohydrates: کمی کمتر از 250 (e.g., 200-249)
    # Fiber: کمی بیشتر از 25 (e.g., 26-35)
    # Calcium: کمی بیشتر از 1000 (e.g., 1001-1200)
    # Iron: کمی بیشتر از 18 (e.g., 18.1-22)
    daily_optimal_ranges = {
        "Calories_kcal": (1800, 1999),
        "Protein_g": (101, 130),  # Giving a bit wider optimal range for protein
        "Fat_g": (50, 60),
        "Carbohydrates_g": (200, 249),
        "Fiber_g": (26, 40),  # Wider optimal for fiber
        "Calcium_mg": (1001, 1500),  # Wider optimal for calcium
        "Iron_mg": (18.1, 25),  # Wider optimal for iron
    }
    # Monthly optimal ranges
    monthly_optimal_ranges = {
        k: (v[0] * 30, v[1] * 30) for k, v in daily_optimal_ranges.items()
    }

    return monthly_min_req, monthly_optimal_ranges


def print_solution_details(
    solution_quantities, food_df, monthly_min_req, monthly_optimal_ranges, budget
):
    if solution_quantities is None:
        print("No solution found.")
        return

    print("\n--- Best Food Plan (Monthly Quantities in kg) ---")
    total_cost = 0
    achieved_nutrients = {col: 0 for col in monthly_min_req.keys()}

    purchased_foods = []
    for i, quantity_kg in enumerate(solution_quantities):
        if quantity_kg > 0.01:  # Consider foods with more than 10g
            food_item = food_df.iloc[i]
            cost_item = quantity_kg * food_item["Price_Toman_per_kg"]
            total_cost += cost_item
            purchased_foods.append(
                {
                    "Food": food_item["Food"],
                    "Quantity_kg": quantity_kg,
                    "Cost_Toman": cost_item,
                }
            )
            for nutrient in achieved_nutrients.keys():
                achieved_nutrients[nutrient] += (
                    quantity_kg * food_item[nutrient]
                )  # food_df nutrients are per kg

    # Print purchased foods
    purchased_df = pd.DataFrame(purchased_foods)
    if not purchased_df.empty:
        print(purchased_df.to_string(index=False))
    else:
        print("No food items selected in the solution (quantities are too small).")

    print(f"\n--- Summary ---")
    print(
        f"Total Estimated Cost: {total_cost:,.0f} Toman (Budget: {budget:,.0f} Toman)"
    )
    if total_cost > budget:
        print(f"WARNING: Over budget by {total_cost - budget:,.0f} Toman")

    print("\n--- Nutritional Analysis (Monthly) ---")
    print(
        f"{'Nutrient':<18} | {'Achieved':>12} | {'Min Req.':>12} | {'Optimal Low':>12} | {'Optimal High':>12} | Status"
    )
    print("-" * 100)
    for nutrient, ach_val in achieved_nutrients.items():
        min_req = monthly_min_req[nutrient]
        opt_low, opt_high = monthly_optimal_ranges[nutrient]
        status = ""
        if ach_val < min_req:
            status = f"BELOW MINIMUM! ({(min_req - ach_val)/30:.1f}/day short)"
        elif ach_val < opt_low:
            status = "Above min, below optimal"
        elif ach_val > opt_high:
            if nutrient in ["Calories_kcal", "Fat_g", "Carbohydrates_g"]:
                status = f"Above optimal (by {(ach_val - opt_high)/30:.1f}/day)"
            else:  # Protein, Fiber, Calcium, Iron
                status = f"Above optimal (likely good)"
        else:  # opt_low <= ach_val <= opt_high
            status = "IN OPTIMAL RANGE"

        print(
            f"{nutrient:<18} | {ach_val:>12,.1f} | {min_req:>12,.1f} | {opt_low:>12,.1f} | {opt_high:>12,.1f} | {status}"
        )

    return achieved_nutrients, total_cost, purchased_df


def plot_ga_progress(best_fitness_history, avg_fitness_history):
    plt.figure(figsize=(12, 5))
    plt.plot(best_fitness_history, label="Best Fitness")
    plt.plot(avg_fitness_history, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Fitness Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_nutrient_profile(
    achieved_nutrients, monthly_min_req, monthly_optimal_ranges, food_df
):
    nutrients_to_plot = list(monthly_min_req.keys())
    achieved_values = np.array(
        [achieved_nutrients.get(n, 0) / 30 for n in nutrients_to_plot]
    )  # Daily
    min_req_values = np.array(
        [monthly_min_req.get(n, 0) / 30 for n in nutrients_to_plot]
    )  # Daily
    opt_low_values = np.array(
        [monthly_optimal_ranges.get(n, (0, 0))[0] / 30 for n in nutrients_to_plot]
    )  # Daily
    opt_high_values = np.array(
        [monthly_optimal_ranges.get(n, (0, 0))[1] / 30 for n in nutrients_to_plot]
    )  # Daily

    n_groups = len(nutrients_to_plot)
    fig, ax = plt.subplots(figsize=(15, 8))
    index = np.arange(n_groups)
    bar_width = 0.2

    rects_achieved = ax.bar(
        index - bar_width, achieved_values, bar_width, label="Achieved (Daily Avg)"
    )
    rects_min = ax.bar(
        index, min_req_values, bar_width, label="Min Required (Daily)", alpha=0.7
    )
    # For optimal range, show as a shaded region or error bars
    # Here, just plotting optimal low and high as lines for simplicity or points
    ax.scatter(
        index + bar_width,
        opt_low_values,
        color="green",
        marker="_",
        s=200,
        label="Optimal Low (Daily)",
    )
    ax.scatter(
        index + bar_width,
        opt_high_values,
        color="lime",
        marker="_",
        s=200,
        label="Optimal High (Daily)",
    )

    # Error bars for optimal range
    # lower_error = achieved_values - opt_low_values
    # upper_error = opt_high_values - achieved_values
    # errors = [np.maximum(0, lower_error), np.maximum(0, upper_error)] # ensure non-negative errors
    # ax.errorbar(index - bar_width, achieved_values, yerr=[achieved_values - opt_low_values, opt_high_values - achieved_values],
    #             fmt='none', ecolor='gray', capsize=5, label='Optimal Range')

    ax.set_xlabel("Nutrient")
    ax.set_ylabel("Amount (Daily Average)")
    ax.set_title("Nutritional Profile of Best Solution (Daily Averages)")
    ax.set_xticks(index)
    ax.set_xticklabels(nutrients_to_plot, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_food_contribution(purchased_df):
    if purchased_df is None or purchased_df.empty:
        print("No purchased food data to plot.")
        return

    plt.figure(figsize=(10, 7))

    # Contribution by cost
    # purchased_df.sort_values('Cost_Toman', ascending=False, inplace=True)
    # plt.bar(purchased_df['Food'][:15], purchased_df['Cost_Toman'][:15]) # Top 15 by cost
    # plt.xlabel("Food Item")
    # plt.ylabel("Cost (Toman)")
    # plt.title("Top Food Items by Cost in the Plan")
    # plt.xticks(rotation=75, ha="right")
    # plt.tight_layout()
    # plt.show()

    # Pie chart by quantity (kg) for top items
    purchased_df_sorted_qty = purchased_df.sort_values("Quantity_kg", ascending=False)
    top_n = 10
    other_qty = purchased_df_sorted_qty["Quantity_kg"][top_n:].sum()

    plot_data_qty = purchased_df_sorted_qty[["Food", "Quantity_kg"]][:top_n]
    if other_qty > 0.01:  # Only add 'Others' if it's significant
        plot_data_qty = pd.concat(
            [
                plot_data_qty,
                pd.DataFrame([{"Food": "Others", "Quantity_kg": other_qty}]),
            ],
            ignore_index=True,
        )

    plt.pie(
        plot_data_qty["Quantity_kg"],
        labels=plot_data_qty["Food"],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title(f"Food Plan Composition by Quantity (kg) - Top {top_n} & Others")
    plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Parameters ---
    LILY_BUDGET_MONTHLY = 4_000_000  # Toman

    # GA Parameters
    POPULATION_SIZE = 150  # Increased population size
    N_GENERATIONS = 300  # Increased generations
    CROSSOVER_RATE = 0.85
    MUTATION_RATE = 0.15  # Per-gene mutation rate
    TOURNAMENT_SIZE = 7
    ELITISM_COUNT = 3
    MAX_KG_PER_FOOD_INIT = 1.5  # Max kg for a single food in initial random population (e.g. 1.5kg for a month)
    MAX_KG_PER_FOOD_MUTATION = (
        3.0  # Max kg for a single food after mutation (e.g. 3kg for a month)
    )

    # --- Load Data and Define Requirements ---
    food_data_df = load_and_preprocess_food_data()
    monthly_min_req, monthly_optimal_ranges = define_requirements()

    # --- Initialize and Run GA ---
    print("Starting Genetic Algorithm for Lily's Diet Plan...")
    ga_optimizer = GeneticAlgorithm(
        food_df=food_data_df,
        monthly_requirements=monthly_min_req,
        monthly_optimal_ranges=monthly_optimal_ranges,
        budget=LILY_BUDGET_MONTHLY,
        population_size=POPULATION_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        n_generations=N_GENERATIONS,
        tournament_size=TOURNAMENT_SIZE,
        elitism_count=ELITISM_COUNT,
        max_kg_per_food_init=MAX_KG_PER_FOOD_INIT,
        max_kg_per_food_mutation=MAX_KG_PER_FOOD_MUTATION,
    )

    best_solution_quantities, best_fitness_hist, avg_fitness_hist = ga_optimizer.run()

    # --- Display Results ---
    if best_solution_quantities:
        achieved_nutrients, total_cost, purchased_foods_df = print_solution_details(
            best_solution_quantities,
            food_data_df,
            monthly_min_req,
            monthly_optimal_ranges,
            LILY_BUDGET_MONTHLY,
        )

        # --- Plotting ---
        plot_ga_progress(best_fitness_hist, avg_fitness_hist)
        if achieved_nutrients:
            plot_nutrient_profile(
                achieved_nutrients,
                monthly_min_req,
                monthly_optimal_ranges,
                food_data_df,
            )
        if purchased_foods_df is not None and not purchased_foods_df.empty:
            plot_food_contribution(purchased_foods_df)

    else:
        print("GA could not find a satisfactory solution.")
