# Project Documentation: Lily's Food Basket Optimization (Simulated Annealing Phase)

## 1. Project Overview 🎯

This project aims to help Lily, a 21-year-old student, optimize her monthly food shopping basket. The goal is to design a 30-day food plan that meets her daily nutritional requirements, stays within her monthly budget of 4,000,000 Toman (25% of her 16 million Toman income), and, for extra credit, achieves "more optimal" levels for certain nutrients. This phase specifically implements the **Simulated Annealing (SA)** algorithm to find this optimal basket.

**Daily Nutritional Minimums:**
* Calories: 2000 kcal
* Protein: 100 g
* Fat: 60 g
* Carbohydrates: 250 g
* Fiber: 25 g
* Calcium: 1000 mg
* Iron: 18 mg

**Monthly needs** are 30 times these daily values.

**Optimization Goals (Extra Points):**
* **Calories**: Slightly less than 2000 kcal/day.
* **Protein**: Slightly more than 100 g/day.
* **Fat**: Exactly 60 g/day or slightly less.
* **Carbohydrates**: Slightly less than 250 g/day.
* **Fiber**: Slightly more than 25 g/day.
* **Calcium**: Slightly more than 1000 mg/day.
* **Iron**: Slightly more than 18 mg/day.

---

## 2. Project Structure 📁

The project is organized as follows:

```text
your_project_root/
├── data/
│   ├── foods.csv           # CSV file with food items, nutritional values, and prices
│   └── best_fit.json       # JSON file with Lily's daily minimum nutritional needs
└── SA/
    ├── main.py             # Main script to run the SA optimization
    └── simu_annealing.py   # Contains the SimulatedAnnealing class
    └── SA_Optimization_Progress.png # Output plot generated after a run
```

---

## 3. Setup Instructions ⚙️

1.  **Python**: Ensure you have Python 3 installed.
2.  **Libraries**: Install the necessary Python libraries:
    ```bash
    pip install pandas matplotlib
    ```
3.  **Data Files**:
    * Place the `foods.csv` file in the `data/` directory. This file should contain a list of food items, their nutritional values per 100g, and price per kg. The expected columns include `Food`, `Calories_kcal`, `Protein_g`, `Fat_g`, `Carbohydrates_g`, `Fiber_g`, `Calcium_mg`, `Iron_mg`, and `Price_Toman_per_kg`.
    * Place the `best_fit.json` file (containing daily nutritional minimums as key-value pairs) in the `data/` directory.

---

## 4. Algorithm: Simulated Annealing (SA) 🔥❄️

The Simulated Annealing algorithm is used to find an optimal shopping basket. It works by:

1.  **Initial State**: Starting with a randomly generated shopping basket (a list of quantities in kg for each food item).
2.  **Evaluation (Energy Function)**: Calculating the "energy" (or cost) of the current basket. A lower energy is better. The energy function penalizes:
    * Exceeding the budget.
    * Not meeting monthly minimum nutritional requirements.
    * Deviating from the "optimal" nutrient target levels (for extra points).
    * It also includes a small factor for the total monetary cost to prefer cheaper solutions if all other criteria are met.
3.  **Neighbor Generation**: Creating a "neighbor" solution by making small random changes to the quantities of one or more food items in the current basket. Quantities are ensured to be non-negative.
4.  **Acceptance Criteria**:
    * If the neighbor solution has lower energy (is better), it's always accepted.
    * If the neighbor solution has higher energy (is worse), it might still be accepted based on a probability. This probability is higher at the beginning (high "temperature") and decreases as the algorithm progresses (temperature "cools"). This helps escape local optima.
5.  **Cooling Schedule**: Gradually reducing the "temperature" according to a cooling rate. The process involves a set number of iterations at each temperature level.
6.  **Termination**: The algorithm stops when the temperature reaches a predefined stopping point or after a maximum number of iterations.

---

## 5. Key Files 📜

* ### `SA/simu_annealing.py`
    This file defines the `SimulatedAnnealing` class, which encapsulates the SA logic.
    * `__init__(...)`: Initializes SA parameters, loads food data, nutritional needs, budget, optimal targets configuration, and sets up the initial solution.
    * `generate_initial_solution()`: Creates a random initial shopping basket.
    * `calculate_nutrients_and_cost(...)`: Computes the total nutrients and monetary cost for a given basket.
    * `calculate_energy(...)`: The core evaluation function that determines the "fitness" or "cost" of a basket.
    * `generate_neighbor_solution(...)`: Modifies the current basket slightly to create a new candidate solution.
    * `acceptance_probability(...)`: Calculates the probability of accepting a new solution, especially a worse one.
    * `run()`: Executes the main SA loop, managing temperature, iterations, and solution updates.

* ### `SA/main.py`
    This is the main executable script.
    * Loads food data from `../data/foods.csv` and daily nutritional needs from `../data/best_fit.json`.
    * Defines the `optimal_targets_config` dictionary, which specifies the desired optimal nutrient levels and the weights for penalizing deviations (for the "extra points" feature).
    * Sets SA parameters (initial temperature, cooling rate, budget, etc.).
    * Instantiates and runs the `SimulatedAnnealing` optimizer.
    * Prints the best shopping basket found, its total cost, and a detailed breakdown of its monthly and average daily nutritional content.
    * Generates and saves plots (`SA_Optimization_Progress.png`) showing the algorithm's progress (energy over time, specific nutrient levels in the best solution over time).

* ### `data/foods.csv`
    A CSV file containing the available food items. Each row represents a food item and should include:
    * `Food`: Name of the food item.
    * `Calories_kcal`: Calories per 100g.
    * `Protein_g`: Protein in grams per 100g.
    * `Fat_g`: Fat in grams per 100g.
    * `Carbohydrates_g`: Carbohydrates in grams per 100g.
    * `Fiber_g`: Fiber in grams per 100g.
    * `Calcium_mg`: Calcium in milligrams per 100g.
    * `Iron_mg`: Iron in milligrams per 100g.
    * `Price_Toman_per_kg`: Price in Toman per kilogram.
    *(The script includes basic NaN filling, but the CSV should ideally be clean.)*

* ### `data/best_fit.json`
    A JSON file specifying Lily's minimum daily nutritional needs. Example:
    ```json
    {
        "Calories": 2000,
        "Protein": 100,
        "Fat": 60,
        "Carbohydrates": 250,
        "Fiber": 25,
        "Calcium": 1000,
        "Iron": 18
    }
    ```

---

## 6. Running the Project ▶️

1.  Navigate to the `SA/` directory in your terminal.
    ```bash
    cd path/to/your_project_root/SA/
    ```
2.  Run the `main.py` script:
    ```bash
    python main.py
    ```

---

## 7. Output 📊

* **Console Output**:
    * Progress messages during the SA run.
    * The final best shopping basket, detailing the quantity (in kg) of each selected food item and its cost.
    * The total monetary cost of the optimized basket.
    * A summary of the nutritional content (total monthly and average daily) of the best basket, compared against minimum requirements and optimal targets.
* **Plot File (`SA_Optimization_Progress.png`)**:
    Generated in the `SA/` directory, this image contains plots illustrating:
    * The evolution of the current solution's energy and the best solution's energy over temperature steps.
    * The evolution of key nutrient levels (e.g., Protein, Calories) in the best-found solution over iterations.
    * The cost of the best-found solution over iterations.

---

## 8. Configuration and Tuning 🛠️

Several aspects of the SA algorithm can be tuned in `SA/main.py` for potentially better results:

* **`optimal_targets_config`**: This dictionary in `main.py` is crucial for the "extra points" feature. It defines:
    * `*_target_daily`: The desired daily level for each nutrient (e.g., `Calories_target_daily: 1900`).
    * `*_weight_*`: Penalty weights associated with deviations from these targets (e.g., `Protein_weight_under`). Fine-tuning these weights will influence how strongly the algorithm prioritizes each optimal target.
* **Simulated Annealing Parameters**:
    * `initial_temp`: Starting temperature. Should be high enough to allow broad exploration.
    * `cooling_rate`: Factor by which temperature is multiplied at each step (e.g., 0.99). Slower cooling (closer to 1) can lead to better solutions but takes longer.
    * `stopping_temp`: Temperature at which the algorithm halts.
    * `max_iter_per_temp`: Number of neighbors generated and evaluated at each temperature level.
* **Energy Function Weights**: Inside `simu_annealing.py`, within the `calculate_energy` method:
    * `budget_overspend_penalty_weight`: Controls the penalty for exceeding the budget.
    * `min_needs_penalty_weights`: A dictionary weighting the penalties for not meeting each minimum nutritional requirement.
    * `cost_weight`: A small factor to make the algorithm prefer cheaper solutions if all other nutritional criteria are met equally well.

Adjusting these parameters and weights may be necessary to achieve the desired balance between meeting all constraints and optimizing for Lily's preferences.
