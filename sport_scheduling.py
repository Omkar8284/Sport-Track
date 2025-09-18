import pandas as pd
import random
from itertools import combinations
from deap import creator, base, tools, algorithms
from datetime import datetime

# Load your actual dataset from a CSV file
df1 = pd.read_csv('IPL_Matches_2022.csv')
df1['Date'] = pd.to_datetime(df1['Date'], format='%d-%m-%Y')

# Sort the DataFrame by the 'Date' column
df = df1.sort_values(by='Date')

# Number of generations and population size for genetic algorithm
num_generations = 100
population_size = 20

# Define genetic algorithm operators
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Function to create an individual (a possible match schedule)
# Function to create an individual (a possible match schedule)
from random import shuffle
def create_individual():
    teams = df['Team1'].tolist()
    venues = df['Venue'].tolist()
    timings = df['Date'].tolist()

    matches = list(combinations(teams, 2))
    random.shuffle(matches)

    schedule = []
    day_matches = []
    for match, venue, timing in zip(matches, venues, timings):
        if len(day_matches) == 2:
            schedule.extend(day_matches)
            day_matches = []

        # Check if a team is not playing against itself
        if match[0] != match[1]:
            day_matches.append({'Match': match, 'Venue': venue, 'Date': timing})

    # Add the remaining matches
    schedule.extend(day_matches)

    # Sort the schedule by ascending date
    schedule.sort(key=lambda x: x['Date'])

    return schedule



toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Function to evaluate the fitness of an individual (schedule)
def evaluate(schedule):
    unique_matches = set()
    repeated_matches_penalty = 0
    preparation_days_penalty = 0
    for i, (match, _, _) in enumerate(schedule):
        if match in unique_matches:
            repeated_matches_penalty += 1
        unique_matches.add(match)

        # Penalize preparation day
        if i > 0:
            previous_date = schedule[i - 1]['Date']
            current_date = schedule[i]['Date']
            if (current_date - previous_date).days == 1:
                preparation_days_penalty += 1

    # 3. Balance the number of home and away matches for each team
    home_away_penalty = 0
    for team in df['Team1']:
        home_matches = [match for match, _, _ in schedule if team in match and df[df['Team1'] == team]['Venue'].iloc[0] in [venue for _, venue, _ in schedule]]
        away_matches = [match for match, _, _ in schedule if team in match and df[df['Team2'] == team]['Venue'].iloc[0] not in [venue for _, venue, _ in schedule]]
        if len(home_matches) != len(away_matches):
            home_away_penalty += abs(len(home_matches) - len(away_matches))

    # Combine penalties and distances to calculate overall fitness
    fitness = -repeated_matches_penalty - preparation_days_penalty - home_away_penalty

    return (fitness,)


# Genetic algorithm
def genetic_algorithm(num_generations, population_size):
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size // 2, lambda_=population_size // 2,
                              cxpb=0.7, mutpb=0.2, ngen=num_generations, stats=None, halloffame=None, verbose=True)
    best_schedule = tools.selBest(population, k=1)[0]
    return best_schedule

# Run the genetic algorithm
best_schedule = genetic_algorithm(num_generations, population_size)

print("Best Match Schedule:")

sorted_schedule = sorted(best_schedule, key=lambda x: x['Date'])
for index, match in enumerate(sorted_schedule, start=1):
    print(f"{index} Match: {match['Match']}, Venue: {match['Venue']}, Date: {match['Date']}")
    