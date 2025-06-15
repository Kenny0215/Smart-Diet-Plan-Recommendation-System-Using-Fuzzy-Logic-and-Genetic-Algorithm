import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

meals_df = pd.read_csv("meal.csv")
meal_ids = meals_df["Meal_ID"].tolist()

# ----------------------
# Fuzzy Logic Functions
# ----------------------

## a) Creation of Membership Functions

def triangular(x, a, b, c):
    """Triangular membership function."""
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

def age_membership(age):
    return {
        "Young": triangular(age, 18, 20, 30),
        "Adult": triangular(age, 29, 35, 45),
        "Elder": triangular(age, 44, 55, 60)
    }

def bmi_membership(bmi):
    return {
        "Underweight": triangular(bmi, 0, 16, 18.5),
        "Normal": triangular(bmi, 18, 22, 25),
        "Overweight": triangular(bmi, 24, 27, 30),
        "Obese": triangular(bmi, 29, 36, 50)
    }

def activity_membership(activity_level):
    return {
        "Low": triangular(activity_level, 1, 2, 4),
        "Moderate": triangular(activity_level, 3, 5, 7),
        "High": triangular(activity_level, 6, 8, 10)
    }

## These membership functions determine the degree to which an input (age, BMI, activity level)
## belongs to each fuzzy category.

def fuzzy_health_assessment(age, bmi, activity, diabetes=False, hypertension=False):

    # Get membership degrees for each input variable
    age_m = age_membership(age)
    bmi_m = bmi_membership(bmi)
    act_m = activity_membership(activity)

    rules_triggered = []
    risk_category = None

    # --- High Risk Rules (Applied in order of priority) ---

    # Rule: IF BMI IS Obese AND Activity IS Low AND Hypertension IS True THEN Risk IS High.
    if bmi_m.get('Obese', 0) > 0 and act_m.get('Low', 0) > 0 and hypertension:
        risk_category = "High"
        rules_triggered.append("Rule: Obese BMI, Low Activity, and Hypertension => High Risk")

    # Rule: IF Age IS Elder AND (Diabetes IS True OR Hypertension IS True) THEN Risk IS High.
    if risk_category is None and age_m.get('Elder', 0) > 0 and (diabetes or hypertension):
        risk_category = "High"
        rules_triggered.append("Rule: Elder Age with Diabetes or Hypertension => High Risk")

    # Rule: if bmi is obese and diabetes is true then risk is high.
    if risk_category is None and bmi_m.get('Obese', 0) > 0 and diabetes:
        risk_category = "High"
        rules_triggered.append("Rule: Obese BMI and Diabetes => High Risk")

    # Rule: IF Activity IS Low AND (Diabetes IS True OR Hypertension IS True) THEN Risk IS High.
    if risk_category is None and act_m.get('Low', 0) > 0 and (diabetes or hypertension):
        risk_category = "High"
        rules_triggered.append("Rule: Low Activity with Diabetes or Hypertension => High Risk")

    # Rule: If age high BMI low , higher risk
    if risk_category is None and age_m.get('Elder', 0) > 0 and bmi_m.get('Underweight', 0) > 0:
        risk_category = "High"
        rules_triggered.append("Rule: Elder Age and Underweight => High Risk")

    # Rule: If age medium BMI low , higher risk
    if risk_category is None and age_m.get('Adult', 0) > 0 and bmi_m.get('Underweight', 0) > 0:
        risk_category = "High"
        rules_triggered.append("Rule: Adult Age and Underweight => High Risk")

    # --- Moderate Risk Rules ---

    # Rule: IF Age IS Adult AND BMI IS Normal AND (Diabetes IS True OR Hypertension IS True) THEN Risk IS Medium.
    if risk_category is None and age_m.get('Adult', 0) > 0 and bmi_m.get('Normal', 0) > 0 and (diabetes or hypertension):
        risk_category = "Moderate"
        rules_triggered.append("Rule: Adult with Normal BMI having Diabetes or Hypertension => Moderate Risk")

    # Rule: IF Age IS Elder AND BMI IS Normal AND Activity IS Moderate THEN Risk IS Medium.
    if risk_category is None and age_m.get('Elder', 0) > 0 and bmi_m.get('Normal', 0) > 0 and act_m.get('Moderate', 0) > 0:
        risk_category = "Moderate"
        rules_triggered.append("Rule: Elder with Normal BMI and Moderate Activity => Moderate Risk")

    # Rule: IF Age IS Young AND Diabetes IS True THEN Risk IS Medium.
    if risk_category is None and age_m.get('Young', 0) > 0 and diabetes:
        risk_category = "Moderate"
        rules_triggered.append("Rule: Young Age with Diabetes => Moderate Risk")

    # Rule: IF Age IS Adult AND BMI IS Overweight THEN Risk IS Medium.
    if risk_category is None and age_m.get('Adult', 0) > 0 and bmi_m.get('Overweight', 0) > 0:
        risk_category = "Moderate"
        rules_triggered.append("Rule: Adult with Overweight BMI => Moderate Risk")

    # Rule: IF BMI IS Underweight AND Activity IS Low THEN Risk IS Medium.
    if risk_category is None and bmi_m.get('Underweight', 0) > 0 and act_m.get('Low', 0) > 0:
        risk_category = "Moderate"
        rules_triggered.append("Rule: Underweight BMI and Low Activity => Moderate Risk")

    # Rule: IF BMI IS Underweight AND Activity IS High THEN Risk IS Low to Medium. (Categorized as Moderate)
    if risk_category is None and bmi_m.get('Underweight', 0) > 0 and act_m.get('High', 0) > 0:
        risk_category = "Moderate"
        rules_triggered.append("Rule: Underweight BMI and High Activity => Moderate Risk")


    # --- Explanation and Return Logic ---

    if risk_category is not None:
        # If a rule was triggered, format the explanation
        explanation = {
            "Risk Category": risk_category,
            "Rules Triggered": rules_triggered,
            "Final Risk Score": "Determined by specific health rules."
        }

    else:
        # If no rules were triggered, use the original score-based calculation
        age_weights = {"Young": 10, "Adult": 40, "Elder": 70}
        bmi_weights = {"Underweight": 60, "Normal": 10, "Overweight": 40, "Obese": 80}
        activity_weights = {"Low": 70, "Moderate": 40, "High": 10}

        # Calculate weighted average for each factor
        age_risk_val = sum(age_m[cat] * age_weights[cat] for cat in age_m) / (sum(age_m.values()) + 1e-6)
        bmi_risk_val = sum(bmi_m[cat] * bmi_weights[cat] for cat in bmi_m) / (sum(bmi_m.values()) + 1e-6)
        act_risk_val = sum(act_m[cat] * activity_weights[cat] for cat in act_m) / (sum(act_m.values()) + 1e-6)

        # Combine scores
        risk_score = (age_risk_val + bmi_risk_val + act_risk_val) / 3

        # Add penalties for pre-existing conditions
        if diabetes:
            risk_score += 15
        if hypertension:
            risk_score += 10

        risk_score = np.clip(risk_score, 0, 100)

        # Determine risk category based on final score
        if risk_score < 40:
            risk_category = "Low"
        elif risk_score < 70:
            risk_category = "Moderate"
        else:
            risk_category = "High"

        explanation = {
            "Age Risk Contribution": f"{round(age_risk_val, 2)}/100",
            "BMI Risk Contribution": f"{round(bmi_risk_val, 2)}/100",
            "Activity Risk Contribution": f"{round(act_risk_val, 2)}/100",
            "Final Risk Score": round(risk_score, 2),
            "Risk Category": risk_category,
            "Rules Triggered": ["Score-based classification (no specific rules met)"]
        }

    return explanation.get("Final Risk Score"), risk_category, explanation


# Get_recommendations to use new rules
def get_recommendations(risk_category, age, bmi, activity, diabetes, hypertension):

    # Get the fuzzy membership degrees for the inputs
    age_m = age_membership(age)
    bmi_m = bmi_membership(bmi)
    act_m = activity_membership(activity)

    recommendations = {}

    # --- Recommendations for LOW risk ---
    if risk_category == "Low":
        # Rule: IF Risk IS Low AND Age IS Elder THEN Recommended_Calcium IS Higher.
        if age_m.get('Elder', 0) > 0:
            recommendations['Calcium Intake'] = 'Higher'

        # Rule: IF Risk IS Low AND BMI IS Underweight THEN Recommended_Calories IS Higher_Moderate AND Recommended_Protein IS Higher_Moderate.
        if bmi_m.get('Underweight', 0) > 0:
            recommendations['Caloric Intake'] = 'Moderate'
            recommendations['Protein Intake'] = 'Moderate'

        # Rule: IF Risk IS Low AND Age IS Young AND Activity IS High THEN Recommended_Calories IS Very_High.
        if age_m.get('Young', 0) > 0 and act_m.get('High', 0) > 0:
            recommendations['Caloric Intake'] = 'High'

    # --- Recommendations for MODERATE risk ---
    elif risk_category == "Moderate":
        # Rule: IF Risk IS Medium AND Hypertension IS True THEN Recommended_Sodium IS Low AND Recommended_Potassium IS Higher.
        if hypertension:
            recommendations['Sodium Intake'] = 'Low'
            recommendations['Potassium Intake'] = 'Higher'

        # Rule: IF Risk IS Medium AND BMI IS Overweight THEN Recommended_Calories IS Lower_Moderate AND Recommended_Fat IS Lower_Moderate.
        if bmi_m.get('Overweight', 0) > 0:
            recommendations['Caloric Intake'] = 'Low'
            recommendations['Fat Intake'] = 'Low'

        # Rule: IF Risk IS Medium AND Age IS Elder THEN Recommended_Fiber IS Higher.
        if age_m.get('Elder', 0) > 0:
            recommendations['Fiber Intake'] = 'High'

    # --- Recommendations for HIGH risk ---
    elif risk_category == "High":
        # Rule: IF Risk IS High AND Diabetes IS True THEN Recommended_Carbs IS Low AND Recommended_Sugar IS Very_Low.
        if diabetes:
            recommendations['Carbohydrate Intake'] = 'Low'
            recommendations['Sugar Intake'] = 'Low'

        # Rule: IF Risk IS High AND Activity IS High THEN Recommended_Protein IS Very_High.
        if act_m.get('High', 0) > 0:
            recommendations['Protein Intake'] = 'High'

    # Add a general recommendation if no specific rules were triggered
    if not recommendations:
        recommendations['General Advice'] = 'Maintain a balanced diet and regular exercise. Consult a healthcare professional for personalized advice.'

    return recommendations

# ---------------------------------
# Genetic Algorithm (GA) Functions
# ---------------------------------
POPULATION_SIZE = 500
NUM_DAYS = 7
MEALS_PER_DAY = 3
CHROMOSOME_LENGTH = NUM_DAYS * MEALS_PER_DAY
GENERATIONS = 50
TARGET_CALORIES = 2100

w_macro = 0.5
w_variety = 0.3
w_allergy = 0.2


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

meals_data = {
    "Meal_ID": list(range(1, 101)),
    "Meal_Name": [f"Meal {i}" for i in range(1, 101)],
    "Calories": [random.randint(300, 800) for _ in range(100)],
    "Protein_g": [random.randint(10, 50) for _ in range(100)],
    "Carbohydrate_g": [random.randint(30, 100) for _ in range(100)],
    "Fat_g": [random.randint(10, 40) for _ in range(100)],
    "Fiber_g": [random.randint(5, 15) for _ in range(100)],
}
meals_df = pd.DataFrame(meals_data)
meal_ids = meals_df["Meal_ID"].tolist()

# Define your GA parameters (should be global or accessible)
POPULATION_SIZE = 500
NUM_DAYS = 7
MEALS_PER_DAY = 3
CHROMOSOME_LENGTH = NUM_DAYS * MEALS_PER_DAY
GENERATIONS = 50
TARGET_CALORIES = 2100

w_macro = 0.5
w_variety = 0.3
w_allergy = 0.2


if "ga_result" not in st.session_state:
    st.session_state.ga_result = None
if "fitness_history" not in st.session_state:
    st.session_state.fitness_history = None


def fitness_function(chromosome, high_protein=True):
    # Ensure meals_df is accessible here.
    # If meals_df is very large, consider optimizing this lookup or passing it.
    plan = meals_df[meals_df["Meal_ID"].isin(chromosome)]
    total_calories = plan["Calories"].sum()
    target_weekly = TARGET_CALORIES * 7
    macro_diff_score = abs(target_weekly - total_calories)
    variety_score = CHROMOSOME_LENGTH - len(set(chromosome))

    penalty = 0
    protein_total = plan["Protein_g"].sum()
    avg_protein = protein_total / CHROMOSOME_LENGTH
    if high_protein:
        if avg_protein < 10:
            penalty += (10 - avg_protein) * 100
        else:
            penalty -= 50

    return w_macro * macro_diff_score + w_variety * variety_score + w_allergy * penalty

def create_chromosome():
    return [random.choice(meal_ids) for _ in range(CHROMOSOME_LENGTH)]

def tournament_selection(population, k=3, high_protein=True):
    selected = random.sample(population, k)
    selected.sort(key=lambda chromo: fitness_function(chromo, high_protein))
    return selected[0]

def crossover(parent1, parent2):
    day = random.randint(0, NUM_DAYS - 1)
    start = day * MEALS_PER_DAY
    end = start + MEALS_PER_DAY
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[start:end], child2[start:end] = parent2[start:end], parent1[start:end]
    return child1, child2

def mutate(chromosome, mutation_rate=0.15):
    new_chromosome = chromosome.copy()
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < mutation_rate:
            new_chromosome[i] = random.choice(meal_ids)
    return new_chromosome

def run_ga(high_protein=True, elitism_size=5):
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    best_chromo = None
    best_fit = float('inf')
    fitness_history = []
    no_improvement = 0

    # --- TUNED GA PARAMETERS ---
    # Set patience to run almost all generations to observe full behavior initially
    # If the graph still flattens, you can try increasing GENERATIONS or other parameters
    patience_tuned = GENERATIONS # Effectively runs for all GENERATIONS before considering early stopping
    min_delta_tuned = 0.001     # Allows even very small improvements to count

    # Try a slightly higher initial mutation rate for more exploration
    initial_mutation_rate_tuned = 0.25 # Increased from 0.15

    # Tournament selection 'k' - you can adjust this if needed
    tournament_k_tuned = 3 # Kept as 3, but can try 2 for less selection pressure

    # --- END TUNED GA PARAMETERS ---

    for generation in range(GENERATIONS):
        # Evaluate fitness for current population
        fitness_scores = [(chromo, fitness_function(chromo, high_protein)) for chromo in population]
        fitness_scores.sort(key=lambda x: x[1])

        # Elitism: preserve top individuals
        elites = [chromo for chromo, _ in fitness_scores[:elitism_size]]
        current_best_chromo_gen, current_best_fit_gen = fitness_scores[0] # Best of the current generation

        # Update the overall best chromosome and fitness
        if current_best_fit_gen < best_fit:
            if (best_fit - current_best_fit_gen) > min_delta_tuned: # Only reset counter if significant improvement
                no_improvement = 0
            best_fit = current_best_fit_gen
            best_chromo = current_best_chromo_gen
        else:
            no_improvement += 1 # No improvement in this generation or improvement was less than min_delta

        # Early stopping
        if no_improvement >= patience_tuned:
            print(f"Early stopping at generation {generation} due to no significant improvement.")
            break

        # Record the overall best fitness found so far
        fitness_history.append(best_fit)

        # Create new population
        new_population = elites.copy()
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, k=tournament_k_tuned, high_protein=high_protein)
            parent2 = tournament_selection(population, k=tournament_k_tuned, high_protein=high_protein)
            child1, child2 = crossover(parent1, parent2)

            # Adaptive mutation: reduce rate over time
            # Ensure adapt_mutation doesn't go below a reasonable minimum if you want continued exploration
            adapt_mutation = initial_mutation_rate_tuned * (1 - generation / GENERATIONS)
            # You might add a lower bound: adapt_mutation = max(0.01, initial_mutation_rate_tuned * (1 - generation / GENERATIONS))

            child1 = mutate(child1, mutation_rate=adapt_mutation)
            child2 = mutate(child2, mutation_rate=adapt_mutation)
            new_population.extend([child1, child2])

        # Ensure population size remains constant
        population = new_population[:POPULATION_SIZE]

    return best_chromo, fitness_history

# -------------------------------
# Streamlit App Layout with Tabs
# -------------------------------

st.set_page_config(page_title="Fuzzy-GA Diet Planner", layout="wide")
st.title("Smart Diet Plan Recommendation Using Fuzzy-GA")

with st.sidebar:
    st.title("Fuzzy-GA Diet Recommendation")
    st.markdown(
        """
        **About:**
        This app uses fuzzy logic to assess your health risk based on your Age, BMI and Activity Level.
        It then provides personalized nutrient recommendations and uses a genetic algorithm (GA)
        to generate an optimized weekly meal plan which suited to your dietary preferences.
        """
    )

# Tabs for Navigation
tabs = st.tabs(["Overview", "Health Assessment", "Meal Plan Optimization", "Membership Functions"])

# ----------------------------
# Tab 1: Overview
# ----------------------------
with tabs[0]:
    st.header("Overview")
    st.markdown(
        """
        **Smart Diet Plan Recommendation** integrates:
        - A **Fuzzy Logic** to evaluate your health risk and provide nutrient recommendations.
        - A **Genetic Algorithm** to optimize a weekly meal plan that fits your nutritional needs and dietary preferences.

        Use the tabs to explore:
        - Your health assessment based on your inputs.
        - The optimized meal plan and detailed nutrient breakdowns.
        - Interactive membership function plots that explain the fuzzy logic.
        """
    )

# ----------------------------
# Tab 2: Health Assessment
# ----------------------------
with tabs[1]:
    st.header("Health Assessment")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enter Your Health Details")
        user_age = st.number_input("Age", min_value=18, max_value=60, value=25, key="age")
        user_bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, step=0.1, key="bmi")
        user_activity = st.slider("Activity Level (1 low - 10 high)", 1, 10, 5, key="activity")
        user_diabetes = st.checkbox("Diabetes", value=False, key="diabetes")
        user_hypertension = st.checkbox("Hypertension", value=False, key="hypertension")
    with col2:
        risk, risk_cat, fuzzy_expl = fuzzy_health_assessment(user_age, user_bmi, user_activity, user_diabetes, user_hypertension)
        recs = get_recommendations(user_age, user_bmi, user_activity, user_diabetes, user_hypertension, risk_cat)
        st.subheader("Your Health Summary")
        st.markdown(f"**Risk Category:** {risk_cat}")
        st.markdown("**Nutrient Recommendations:**")
        st.write(recs)
    st.markdown("---")
    st.markdown("**Fuzzy Logic Explanation:**")
    st.json(fuzzy_expl)

# ----------------------------
# Tab 3: Meal Plan Optimization
# ----------------------------
with tabs[2]:
    st.header("Meal Plan Optimization")
    st.markdown("Click the button below to generate an optimized weekly meal plan focusing on high protein intake.")

    if st.button("Generate Meal Plan"):
        with st.spinner("Optimizing your meal plan..."):
            best_plan, fitness_history = run_ga(high_protein=True)
            st.session_state.ga_result = best_plan
            st.session_state.fitness_history = fitness_history
        st.success("Meal plan generated!")

    if st.session_state.ga_result is not None:
        # --- Display GA Evolution Chart ---
        st.subheader("GA Evolution")
        fig_ga, ax_ga = plt.subplots(figsize=(8, 4))
        ax_ga.plot(st.session_state.fitness_history, marker='o', linestyle='-')
        ax_ga.set_xlabel("Generation")
        ax_ga.set_ylabel("Best Fitness")
        ax_ga.set_title("GA Evolution: Fitness over Generations")
        st.pyplot(fig_ga)

        # --- Convert Meal IDs to Meal Names ---
        meal_names = []
        for meal_id in st.session_state.ga_result:
             meal_row = meals_df[meals_df["Meal_ID"] == meal_id]
        if not meal_row.empty:
             name = meal_row.iloc[0]["Meal_Name"]
             meal_names.append(f"Meal {meal_id} - {name}")  # New: "Meal ID - Name"
        else:
             meal_names.append(f"Meal {meal_id} - Unknown")

        # --- Reshape into Weekly Plan ---
        plan_df = pd.DataFrame(
            np.array(meal_names).reshape(NUM_DAYS, MEALS_PER_DAY),
            columns=["Breakfast", "Lunch", "Dinner"]
        )

        st.subheader("Optimized Weekly Meal Plan")
        st.dataframe(plan_df)

        # --- Daily Nutrient Totals ---
        daily_totals = []
        for day in range(NUM_DAYS):
            day_ids = st.session_state.ga_result[day * MEALS_PER_DAY: (day + 1) * MEALS_PER_DAY]
            day_plan = meals_df[meals_df["Meal_ID"].isin(day_ids)]
            totals = {
                "Calories": day_plan["Calories"].sum(),
                "Protein (g)": day_plan["Protein_g"].sum(),
                "Carbs (g)": day_plan["Carbohydrate_g"].sum(),
                "Fat (g)": day_plan["Fat_g"].sum(),
                "Fiber (g)": day_plan["Fiber_g"].sum()
            }
            daily_totals.append(totals)

        nutrient_df = pd.DataFrame(daily_totals, index=[f"Day {i+1}" for i in range(NUM_DAYS)])
        st.subheader("Daily Nutrient Breakdown")
        st.dataframe(nutrient_df)

        # --- Interactive Meal Nutrient Breakdown ---
        st.subheader("View Meal Nutrient Breakdown")
        meal_choice = st.selectbox(
            "Select a Meal from the Optimized Plan",
            options=st.session_state.ga_result,
            format_func=lambda x: f"{meals_df[meals_df['Meal_ID'] == x].iloc[0]['Meal_Name']} ({x})",
            key="meal_select"
        )
        selected_meal = meals_df[meals_df["Meal_ID"] == meal_choice].iloc[0]

        fig_meal, ax_meal = plt.subplots(figsize=(8, 4))
        meal_nutrients = ["Calories", "Protein_g", "Carbohydrate_g", "Fat_g", "Fiber_g"]
        nutrient_values = [selected_meal[n] for n in meal_nutrients]
        ax_meal.bar(meal_nutrients, nutrient_values, color="skyblue")
        ax_meal.set_ylabel("Value")
        ax_meal.set_title(f"Nutrient Breakdown: {selected_meal['Meal_Name']} ({meal_choice})")
        st.pyplot(fig_meal)

    else:
        st.info("Click 'Generate Meal Plan' to optimize your weekly meal plan.")

# ----------------------------
# Tab 4: Membership Functions
# ----------------------------
with tabs[3]:
    st.header("Membership Function Visualizations")

    x_age = np.linspace(15, 65, 500)
    x_bmi = np.linspace(10, 50, 1000)
    x_activity = np.linspace(1, 10, 500)

    fig1, ax1 = plt.subplots()
    ax1.plot(x_age, triangular(x_age, 18, 20, 30), label="Young")
    ax1.plot(x_age, triangular(x_age, 29, 35, 45), label="Adult")
    ax1.plot(x_age, triangular(x_age, 44, 55, 60), label="Elder")
    ax1.set_title("Age Membership Functions")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.plot(x_bmi, triangular(x_bmi, 0, 16, 18.5), label="Underweight")
    ax2.plot(x_bmi, triangular(x_bmi, 18, 22, 25), label="Normal")
    ax2.plot(x_bmi, triangular(x_bmi, 24, 27, 30), label="Overweight")
    ax2.plot(x_bmi, triangular(x_bmi, 29, 36, 50), label="Obese")
    ax2.set_title("BMI Membership Functions")
    ax2.legend()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(x_activity, triangular(x_activity, 1, 2, 4), label="Low")
    ax3.plot(x_activity, triangular(x_activity, 3, 5, 7), label="Moderate")
    ax3.plot(x_activity, triangular(x_activity, 6, 8, 10), label="High")
    ax3.set_title("Activity Level Membership Functions")
    ax3.legend()
    st.pyplot(fig3)
