# Lily's Diet Optimization using Genetic Algorithm

This project implements a Genetic Algorithm (GA) to help Lily, a 21-year-old computer science student, optimize her monthly food shopping list. The goal is to create a 30-day food plan that meets her daily nutritional requirements, stays within her budget, and aims for optimal nutrient levels for better health (bonus feature).

## Project Overview

Lily has a monthly budget of 4,000,000 Toman for groceries. She has identified her daily nutritional needs and has a list of common food items with their nutritional information and prices. This program uses a Genetic Algorithm to find a combination and quantity of these food items that:

1.  Satisfies her minimum monthly nutritional needs.
2.  Does not exceed her monthly budget of 4,000,000 Toman.
3.  Optimizes the intake of certain nutrients (e.g., slightly less than 2000 kcal, more protein, etc.) as per her preferences for a healthier diet.

## Features

- **Genetic Algorithm Core**: Implements standard GA operators:
  - **Chromosome Representation**: A list of quantities (in kg) for each available food item for a 30-day period.
  - **Initialization**: Creates an initial population of random food plans.
  - **Fitness Function**: Evaluates each food plan based on:
    - Adherence to budget (heavy penalty for exceeding).
    - Meeting minimum monthly nutritional requirements (heavy penalty for deficiencies).
    - Achieving optimal nutrient ranges (bonuses for being within, fine-tuned penalties for being outside optimal but above minimum).
  - **Selection**: Tournament selection to choose parents for the next generation.
  - **Crossover**: Arithmetic/Blend crossover to combine parent solutions.
  - **Mutation**: Randomly alters food quantities in a plan to introduce diversity, ensuring quantities remain non-negative and within reasonable limits.
  - **Elitism**: Preserves the best solutions from one generation to the next.
- **Data Driven**: Reads food nutritional data and prices from `foods.csv`.
- **Customizable Parameters**: GA parameters (population size, generations, mutation rate, etc.) can be adjusted in `main.py`.
- **Detailed Output**:
  - Prints the best food plan found, including quantities of each food and their cost.
  - Provides a summary of total cost and a detailed nutritional analysis (achieved vs. required vs. optimal).
- **Visualizations**:
  - Plots the best and average fitness scores over generations.
  - Displays the nutritional profile of the best solution compared to targets.
  - Shows a pie chart of the food composition in the best plan by quantity.

## File Structure

```
├── GA/
│   ├── genetic_algo.py # Contains the GeneticAlgorithm class implementation
│   ├── main.py # Main script to run the optimization, define requirements, and display results
│   └── README.md # This documentation file
└── data/
    └── foods.csv # CSV file containing food data (nutrients per 100g, price per kg)
```

## Prerequisites

- Python 3.7+
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`

## Setup and Usage

1.  **Clone the repository (or download the files):**

    ```bash
    # If it were a git repo:
    # git clone <repository_url>
    # cd <repository_directory>
    ```

    Ensure `genetic_algo.py` and `main.py` are in the same directory.
    Ensure `data/foods.csv` is in the same directory.

2.  **Install required libraries:**

    ```bash
    pip install pandas numpy matplotlib
    ```

3.  **Run the program:**
    ```bash
    python main.py
    ```

## How It Works

### 1. Data Loading and Preprocessing (`main.py`)

- The `foods.csv` file is loaded using pandas.
- Nutritional values, initially provided per 100g, are converted to be per kg to align with food prices (per kg) and the GA's food quantity representation (kg).

### 2. Defining Requirements (`main.py`)

- **Minimum Daily Requirements**: Defined for Calories, Protein, Fat, Carbohydrates, Fiber, Calcium, and Iron. These are then scaled to monthly minimums (30 days).
- **Optimal Daily Ranges**: Defined for the same nutrients, specifying preferred lower and upper bounds for a healthier diet. These are also scaled to monthly optimal ranges.
- **Budget**: Lily's monthly budget is set (4,000,000 Toman).

### 3. Genetic Algorithm (`genetic_algo.py`)

The `GeneticAlgorithm` class manages the evolutionary process:

- **Individual (Chromosome)**: A list of floating-point numbers, where each number represents the quantity (in kilograms) of a specific food item to be purchased for the month. The length of the list corresponds to the number of food items available in `foods.csv`.

- **Initialization**: A population of random individuals is created. Each food quantity in an individual is a random value between 0 and `max_kg_per_food_init`.

- **Fitness Evaluation**:
  The `_fitness` method calculates a score for each individual (food plan):

  1.  It calculates the total cost and the total amount of each nutrient provided by the plan.
  2.  **Budget Penalty**: If `total_cost > budget`, a significant penalty is applied, proportional to the amount over budget.
  3.  **Minimum Nutrient Penalty**: For each nutrient, if `achieved_nutrient < monthly_minimum_requirement`, a significant penalty is applied, proportional to the deficit.
  4.  **Optimal Nutrient Scoring (Bonus Feature)**:
      - If a nutrient's achieved value falls within its `monthly_optimal_range`, a bonus is added to the fitness.
      - If a nutrient's achieved value is above the minimum but outside the optimal range (either too low or too high), smaller, fine-tuned penalties are applied. These penalties are weighted based on the nutrient (e.g., exceeding optimal calories is penalized more than exceeding optimal protein).

- **Selection**: Parents for the next generation are chosen using tournament selection. A small group of individuals is randomly selected, and the one with the highest fitness becomes a parent.

- **Crossover**: Selected parent pairs produce offspring using arithmetic/blend crossover. A random blending factor (alpha) determines how genetic material (food quantities) is mixed. Offspring quantities are ensured to be non-negative.

- **Mutation**: Each gene (food quantity) in an offspring has a `mutation_rate` chance of being mutated. Mutation involves replacing the quantity with a new random value between 0 and `max_kg_per_food_mutation`, or adding a small Gaussian noise. This helps maintain diversity and explore new solution spaces. Quantities are capped and ensured non-negative.

- **Elitism**: A small number of the best individuals from the current generation are directly carried over to the next generation, ensuring that good solutions are not lost.

- **Termination**: The algorithm runs for a predefined number of `n_generations`.

### 4. Output and Visualization (`main.py`)

- After the GA completes, the best solution (food plan) found is displayed.
- **Console Output**:
  - A list of purchased foods with their monthly quantities (kg) and estimated costs.
  - The total estimated cost of the plan compared to the budget.
  - A detailed nutritional summary showing achieved monthly values for each nutrient versus the minimum requirements and optimal ranges, along with a status (e.g., "IN OPTIMAL RANGE", "BELOW MINIMUM!").
- **Plots (using Matplotlib)**:
  - **GA Fitness Progression**: Line plot showing the best and average fitness scores of the population across generations. This helps visualize the GA's convergence.
  - **Nutritional Profile**: Bar chart comparing the daily average achieved nutrients from the best plan against daily minimum requirements and optimal range boundaries.
  - **Food Plan Composition**: Pie chart illustrating the proportion of different food items (by quantity in kg) in the best plan.

## Configuration

Key parameters for the Genetic Algorithm can be adjusted in the `main.py` script:

- `LILY_BUDGET_MONTHLY`: Lily's total monthly budget.
- `POPULATION_SIZE`: Number of individuals in each generation.
- `N_GENERATIONS`: Total number of generations the GA will run.
- `CROSSOVER_RATE`: Probability of two parents performing crossover.
- `MUTATION_RATE`: Probability of a gene mutating in an offspring.
- `TOURNAMENT_SIZE`: Number of individuals competing in each selection tournament.
- `ELITISM_COUNT`: Number of best individuals to carry over to the next generation.
- `MAX_KG_PER_FOOD_INIT`: Maximum quantity (kg) for a single food item in the initial random population.
- `MAX_KG_PER_FOOD_MUTATION`: Maximum quantity (kg) for a single food item after a mutation event.

Daily nutritional requirements and optimal ranges are also defined in `main.py` within the `define_requirements` function.

## Interpreting the Results

- **Fitness Plot**: Ideally, the best fitness should increase over generations, and the average fitness should also trend upwards, indicating the population is improving. The gap between best and average fitness can give insights into population diversity.
- **Solution Details**:
  - Check if the **Total Estimated Cost** is within Lily's budget.
  - Review the **Nutritional Analysis**. Ensure all minimums are met ("BELOW MINIMUM!" should not appear for any nutrient).
  - See how many nutrients fall "IN OPTIMAL RANGE". This is the target for the bonus optimization.
- **Food Plan**: The list of foods and quantities provides a practical shopping list. It's important to consider if the quantities are reasonable for consumption (e.g., not too much of one specific item unless intended).

Due to the stochastic nature of Genetic Algorithms, running the program multiple times might yield slightly different (but hopefully similarly good) solutions.

## Potential Future Enhancements

- Implement the **Simulated Annealing** algorithm for comparison or as a hybrid approach.
- Introduce constraints for food variety or maximum/minimum servings of certain food groups.
- Add a Graphical User Interface (GUI) for easier interaction.
- Incorporate food preparation waste or yield factors.
- Allow for more complex dietary preferences (e.g., vegan, gluten-free by filtering `foods.csv`).
