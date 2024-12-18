import pandas as pd
import numpy as np

# Define the roles for the team composition
roles = ['Top', 'Jungle', 'Middle', 'ADC', 'Support']
roles_final_players = ['Top', 'Jng', 'Mid', 'Bot', 'Sup']


# For ChampionStats.csv: Function to normalize the relevant metrics by games played (GP)
def normalized_evaluation_metric(row):
    gp = row['GP'] if row['GP'] != 0 else 1  # Avoid division by zero
    kills = row['K'] / gp
    assists = row['A'] / gp
    deaths = row['D'] / gp
    gd10 = row['GD10'] / gp
    xpd10 = row['XPD10'] / gp
    return kills + assists + gd10 + xpd10 - deaths


# Simulated annealing function for ChampionStats.csv with normalized metrics
def simulated_annealing_normalized(champions_df, roles, max_iter=1000, initial_temp=100.0, cooling_rate=0.95):
    # Filter data by each role
    role_data = {role: champions_df[champions_df['Pos'].str.lower() == role.lower()] for role in roles}

    # Initialize current solution with a random selection from each role ensuring unique champions
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

    # Check if all roles are filled
    if len(current_solution) < len(roles):
        print("Not all roles could be filled. Adjust your data or selection criteria.")
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


# Example function for Final Players df without normalization
def evaluation_metric_final_players(row):
    # Define the evaluation metric for the 'Final players df' dataset
    return row['kills'] + row['assists'] + row['golddiffat15'] + row['golddiffat10'] - row['deaths']


def simulated_annealing_final_players(champions_df, roles, max_iter=1000, initial_temp=100.0, cooling_rate=0.95):
    # Filter data by each role using shortened role names
    role_data = {role: champions_df[champions_df['position'].str.lower() == role.lower()] for role in roles}

    selected_champions = set()
    current_solution = {}
    for role in roles:
        available_champions = role_data[role][~role_data[role]['champion'].isin(selected_champions)]
        if available_champions.empty:
            print(f"No available champions for role {role}. Skipping.")
            continue  # Skip if no available champions
        chosen = available_champions.sample(1).iloc[0]
        current_solution[role] = chosen
        selected_champions.add(chosen['champion'])

    # Check if all roles are filled
    if len(current_solution) < len(roles):
        print("Not all roles could be filled. Adjust your data or selection criteria.")
        return current_solution, 0

    current_value = sum([current_solution[role]['metric'] for role in current_solution if role in current_solution])

    best_solution = current_solution.copy()
    best_value = current_value

    temp = initial_temp

    for i in range(max_iter):
        random_role = np.random.choice(roles)
        available_champions = role_data[random_role][~role_data[random_role]['champion'].isin(selected_champions)]

        if available_champions.empty:
            continue

        new_champion = available_champions.sample(1).iloc[0]

        new_solution = current_solution.copy()
        new_solution[random_role] = new_champion
        new_value = sum([new_solution[role]['metric'] for role in new_solution if role in new_solution])

        delta = new_value - current_value
        acceptance_probability = np.exp(delta / temp) if delta < 0 else 1

        if np.random.rand() < acceptance_probability:
            if random_role in current_solution:
                selected_champions.discard(current_solution[random_role]['champion'])
            current_solution = new_solution
            current_value = new_value
            selected_champions.add(new_champion['champion'])

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

        # Handle NaN values (either drop or fill them)
        df.fillna(0, inplace=True)  # Replace NaN with 0
        # or
        # df.dropna(inplace=True)  # Drop rows with NaN values

        # Apply normalization for ChampionStats.csv
        df['normalized_metric'] = df.apply(normalized_evaluation_metric, axis=1)

        best_solution, _ = simulated_annealing_normalized(df, roles)

    elif dataset_type == 'final_players':
        # Filter to only include winners
        df = df[df['result'] == 1]

        # Convert relevant columns to numeric
        df['kills'] = pd.to_numeric(df['kills'], errors='coerce')
        df['assists'] = pd.to_numeric(df['assists'], errors='coerce')
        df['deaths'] = pd.to_numeric(df['deaths'], errors='coerce')
        df['golddiffat15'] = pd.to_numeric(df['golddiffat15'], errors='coerce')
        df['golddiffat10'] = pd.to_numeric(df['golddiffat10'], errors='coerce')

        # Handle NaN values (either drop or fill them)
        df.fillna(0, inplace=True)  # Replace NaN with 0
        # or
        # df.dropna(inplace=True)  # Drop rows with NaN values

        # Apply a different evaluation metric for Final Players df
        df['metric'] = df.apply(evaluation_metric_final_players, axis=1)
        best_solution, _ = simulated_annealing_final_players(df, roles_final_players)

    # Ensure that the solution covers all roles
    if dataset_type == 'final_players' and len(best_solution) < len(roles_final_players):
        print("Incomplete solution. Not all roles are filled.")
        return pd.DataFrame(columns=['Role', 'Champion'])
    if dataset_type == 'champion_stats' and len(best_solution) < len(roles):
        print("Incomplete solution. Not all roles are filled.")
        return pd.DataFrame(columns=['Role', 'Champion'])

    # Extract the champion names and roles
    solution = [
        (role, best_solution[role]['Champion'] if dataset_type == 'champion_stats' else best_solution[role]['champion'])
        for role in (roles_final_players if dataset_type == 'final_players' else roles)]

    # Create DataFrame to display in table format
    solution_df = pd.DataFrame(solution, columns=['Role', 'Champion'])

    return solution_df


# Example usage
file_path_champion_stats = 'ChampionStats2.csv'
file_path_final_players = 'Final players df.csv'

# Run simulated annealing for ChampionStats.csv
best_solution_df_champion_stats = run_simulated_annealing_on_dataset(file_path_champion_stats,
                                                                     dataset_type='champion_stats')
print("Optimal Team Composition for 'ChampionStats2.csv':")
print(best_solution_df_champion_stats)

# Run simulated annealing for Final Players df.csv
best_solution_df_final_players = run_simulated_annealing_on_dataset(file_path_final_players,
                                                                    dataset_type='final_players')
print("\nOptimal Team Composition for 'Final players df.csv':")
print(best_solution_df_final_players)
