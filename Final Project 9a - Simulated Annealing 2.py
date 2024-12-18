import pandas as pd
import numpy as np

# Define the roles for the team composition
roles = ['Top', 'Jungle', 'Middle', 'ADC', 'Support']
roles_final_players = ['Top', 'Jng', 'Mid', 'Bot', 'Sup']

# Define role constraints
role_constraints = {
    'Top': ['Aatrox', 'Akali','Cassiopeia','Camille', 'Darius', 'Gnar', 'Irelia', 'Jax', 'Malphite', 'Ornn', 'Renekton', 'Sett'],
    'Jungle': ['Elise','Zyra','Brand', 'Evelynn', 'Gragas', 'Hecarim', 'Jarvan IV', 'Karthus', 'Lee Sin', 'Nidalee', 'Olaf', 'RekSai'],
    'Middle': ['Ahri', 'Akali', 'Anivia', 'Azir', 'Cassiopeia', 'Galio', 'Kassadin', 'LeBlanc', 'Orianna', 'Syndra'],
    'ADC': ['Ashe', 'Caitlyn', 'Draven', 'Ezreal', 'Jhin', 'KaiSa', 'Miss Fortune', 'Sivir', 'Tristana', 'Xayah'],
    'Support': ['Alistar', 'Braum', 'Janna', 'Leona', 'Lulu', 'Nautilus', 'Rakan', 'Thresh', 'Yuumi', 'Zyra']
}

# Function to normalize the relevant metrics by games played (GP) including additional metrics
def normalized_evaluation_metric_combined(row):
    gp = row['GP'] if row['GP'] != 0 else 1  # Avoid division by zero
    kills = row['K'] / gp
    assists = row['A'] / gp
    deaths = row['D'] / gp
    gd10 = pd.to_numeric(row['GD10'], errors='coerce') / gp
    xpd10 = pd.to_numeric(row['XPD10'], errors='coerce') / gp
    cspm = row['CSPM']
    dmg_pct = row['DMG%']
    gold_pct = row['GOLD%']
    return kills + assists + gd10 + xpd10 + cspm + dmg_pct + gold_pct - deaths

# Simulated annealing function for ChampionStats.csv with normalized metrics and role constraints
def simulated_annealing_with_constraints(champions_df, roles, role_constraints, max_iter=1000, initial_temp=100.0, cooling_rate=0.95):
    role_data = {
        role: champions_df[champions_df['Champion'].isin(role_constraints[role])]
        for role in roles
    }

    selected_champions = set()
    current_solution = {}
    for role in roles:
        available_champions = role_data[role][~role_data[role]['Champion'].isin(selected_champions)]
        if available_champions.empty:
            print(f"No available champions for role {role}. Skipping.")
            continue
        chosen = available_champions.sample(1).iloc[0]
        current_solution[role] = chosen
        selected_champions.add(chosen['Champion'])

    if len(current_solution) < len(roles):
        print("Not all roles could be filled. Adjust your constraints or data.")
        return current_solution, 0

    current_value = sum(
        [current_solution[role]['normalized_metric'] for role in current_solution if role in current_solution])

    best_solution = current_solution.copy()
    best_value = current_value

    temp = initial_temp

    for i in range(max_iter):
        random_role = np.random.choice(roles)
        available_champions = role_data[random_role][~role_data[random_role]['Champion'].isin(selected_champions)]

        if available_champions.empty:
            continue

        new_champion = available_champions.sample(1).iloc[0]

        new_solution = current_solution.copy()
        new_solution[random_role] = new_champion
        new_value = sum([new_solution[role]['normalized_metric'] for role in new_solution if role in new_solution])

        delta = new_value - current_value
        acceptance_probability = np.exp(delta / temp) if delta < 0 else 1

        if np.random.rand() < acceptance_probability:
            if random_role in current_solution:
                selected_champions.discard(current_solution[random_role]['Champion'])
            current_solution = new_solution
            current_value = new_value
            selected_champions.add(new_champion['Champion'])

        if current_value > best_value:
            best_solution = current_solution
            best_value = current_value

        temp *= cooling_rate

    return best_solution, best_value

# Example function to run simulated annealing on a specified dataset
def run_simulated_annealing_on_dataset(file_path, dataset_type='champion_stats'):
    df = pd.read_csv(file_path)

    if dataset_type == 'champion_stats':
        # Convert relevant columns to numeric, coerce errors to NaN (you can handle them later)
        df['K'] = pd.to_numeric(df['K'], errors='coerce')
        df['A'] = pd.to_numeric(df['A'], errors='coerce')
        df['D'] = pd.to_numeric(df['D'], errors='coerce')
        df['GD10'] = pd.to_numeric(df['GD10'], errors='coerce')
        df['XPD10'] = pd.to_numeric(df['XPD10'], errors='coerce')
        df['GP'] = pd.to_numeric(df['GP'], errors='coerce')
        df['CSPM'] = pd.to_numeric(df['CSPM'], errors='coerce')
        df['DMG%'] = df['DMG%'].str.replace('%', '').astype(float)
        df['GOLD%'] = df['GOLD%'].str.replace('%', '').astype(float)

        # Handle NaN values (either drop or fill them)
        df.fillna(0, inplace=True)

        # Apply the combined normalization for ChampionStats.csv
        df['normalized_metric'] = df.apply(normalized_evaluation_metric_combined, axis=1)

        best_solution, _ = simulated_annealing_with_constraints(df, roles, role_constraints)

    elif dataset_type == 'final_players':
        # This part of the code would handle the 'final_players' dataset similarly
        pass  # As per your use case, you'd implement a similar function for final players

    # Ensure that the solution covers all roles
    if len(best_solution) < len(roles):
        print("Incomplete solution. Not all roles are filled.")
        return pd.DataFrame(columns=['Role', 'Champion'])

    # Extract the champion names and roles
    solution = [(role, best_solution[role]['Champion']) for role in roles]

    # Create DataFrame to display in table format
    solution_df = pd.DataFrame(solution, columns=['Role', 'Champion'])

    return solution_df

# Example usage
file_path_champion_stats = 'ChampionStats2.csv'

# Run simulated annealing for ChampionStats.csv with constraints
best_solution_df_champion_stats = run_simulated_annealing_on_dataset(file_path_champion_stats, dataset_type='champion_stats')
print("Optimal Team Composition for 'ChampionStats2.csv':")
print(best_solution_df_champion_stats)
