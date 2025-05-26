# Lily's Grocery Shopping Cart Optimization Project

## Contributors

**Arman Akhoundy** ([ArmanX5](https://github.com/ArmanX5)) & **Amirreza Abbasian** ([Amirreza0924](https://github.com/Amirreza0924))

## 1. Project Overview 🎯

This project aims to help Lily, a 21-year-old student, optimize her monthly grocery shopping cart. The goal is to design a 30-day food plan that meets her daily nutritional requirements, stays within her monthly budget of 4,000,000 Toman, and, ideally, achieves "more optimal" levels for certain nutrients. The project explores two optimization algorithms: **Genetic Algorithm (GA)** and **Simulated Annealing (SA)**.

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
│   └── best_fit.json       # JSON file with Lily's daily minimum nutritional needs (used in SA)
├── GA/
│   ├── genetic_algo.py     # Contains the GeneticAlgorithm class implementation
│   ├── main.py             # Main script to run the GA optimization and display results
│   └── README.md           # Documentation for the Genetic Algorithm implementation
└── SA/
    ├── main.py             # Main script to run the SA optimization
    ├── simu_annealing.py   # Contains the SimulatedAnnealing class
    └── README.md           # Documentation for the Simulated Annealing implementation
```

---

## 3. Setup Instructions ⚙️

1.  **Python**: Ensure you have Python 3 installed.
2.  **Libraries**: Install the necessary Python libraries:
    ```bash
    pip install pandas matplotlib numpy
    ```
3.  **Data Files**:
    *   Place the `foods.csv` file in the `data/` directory. This file should contain a list of food items, their nutritional values per 100g, and price per kg. The expected columns include `Food`, `Calories_kcal`, `Protein_g`, `Fat_g`, `Carbohydrates_g`, `Fiber_g`, `Calcium_mg`, `Iron_mg`, and `Price_Toman_per_kg`.
    *   Place the `best_fit.json` file (containing daily nutritional minimums as key-value pairs) in the `data/` directory. This file is specifically used by the Simulated Annealing algorithm.

---

## 4. Algorithms Implemented 🤖

### 4.1. Genetic Algorithm (GA)

The Genetic Algorithm, implemented in the `GA/` directory, works by:

1.  **Initialization**: Creating an initial population of random food plans (chromosomes). Each food plan consists of quantities (in kg) for each available food item.
2.  **Fitness Evaluation**: Evaluating each food plan based on its cost, adherence to minimum monthly nutritional requirements, and proximity to optimal nutrient ranges.
3.  **Selection**: Selecting parent food plans for reproduction using tournament selection.
4.  **Crossover**: Creating offspring food plans by combining the quantities from parent plans using arithmetic/blend crossover.
5.  **Mutation**: Introducing random changes to food quantities in the offspring plans to maintain diversity.
6.  **Elitism**: Preserving the best food plans from one generation to the next.
7.  **Termination**: Repeating steps 2-6 for a predefined number of generations.

### 4.2. Simulated Annealing (SA)

The Simulated Annealing algorithm, implemented in the `SA/` directory, works by:

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

## 5. Running the Project ▶️

To run either the GA or SA implementation:

1.  Navigate to the respective directory (`GA/` or `SA/`) in your terminal.
    ```bash
    cd path/to/your_project_root/GA/   # For Genetic Algorithm
    # OR
    cd path/to/your_project_root/SA/   # For Simulated Annealing
    ```
2.  Run the `main.py` script:
    ```bash
    python main.py
    ```

---

## 6. Output 📊

Both algorithms provide detailed output to the console, including:

*   The best shopping basket found, detailing the quantity (in kg) of each selected food item and its cost.
*   The total monetary cost of the optimized basket.
*   A summary of the nutritional content (total monthly and average daily) of the best basket, compared against minimum requirements and optimal targets.

The GA implementation also generates plots illustrating the algorithm's progress and the nutritional profile of the best solution. The SA implementation generates plots showing the energy over time and nutrient levels in the best solution.

---

## 7. Configuration and Tuning 🛠️

### 7.1. Genetic Algorithm

Key parameters for the Genetic Algorithm can be adjusted in the `GA/main.py` script:

*   `LILY_BUDGET_MONTHLY`: Lily's total monthly budget.
*   `POPULATION_SIZE`: Number of individuals in each generation.
*   `N_GENERATIONS`: Total number of generations the GA will run.
*   `CROSSOVER_RATE`: Probability of two parents performing crossover.
*   `MUTATION_RATE`: Probability of a gene mutating in an offspring.
*   `TOURNAMENT_SIZE`: Number of individuals competing in each selection tournament.
*   `ELITISM_COUNT`: Number of best individuals to carry over to the next generation.
*   `MAX_KG_PER_FOOD_INIT`: Maximum quantity (kg) for a single food item in the initial random population.
*   `MAX_KG_PER_FOOD_MUTATION`: Maximum quantity (kg) for a single food item after a mutation event.

Daily nutritional requirements and optimal ranges are also defined in `GA/main.py` within the `define_requirements` function.

### 7.2. Simulated Annealing

Several aspects of the SA algorithm can be tuned in `SA/main.py` for potentially better results:

*   `optimal_targets_config`: This dictionary in `SA/main.py` is crucial for the "extra points" feature. It defines:
    *   `*_target_daily`: The desired daily level for each nutrient (e.g., `Calories_target_daily: 1900`).
    *   `*_weight_*`: Penalty weights associated with deviations from these targets (e.g., `Protein_weight_under`). Fine-tuning these weights will influence how strongly the algorithm prioritizes each optimal target.
*   Simulated Annealing Parameters:
    *   `initial_temp`: Starting temperature. Should be high enough to allow broad exploration.
    *   `cooling_rate`: Factor by which temperature is multiplied at each step (e.g., 0.99). Slower cooling (closer to 1) can lead to better solutions but takes longer.
    *   `stopping_temp`: Temperature at which the algorithm halts.
    *   `max_iter_per_temp`: Number of neighbors generated and evaluated at each temperature level.
*   Energy Function Weights: Inside `SA/simu_annealing.py`, within the `calculate_energy` method:
    *   `budget_overspend_penalty_weight`: Controls the penalty for exceeding the budget.
    *   `min_needs_penalty_weights`: A dictionary weighting the penalties for not meeting each minimum nutritional requirement.
    *   `cost_weight`: A small factor to make the algorithm prefer cheaper solutions if all other nutritional criteria are met equally well.

Adjusting these parameters and weights may be necessary to achieve the desired balance between meeting all constraints and optimizing for Lily's preferences.
