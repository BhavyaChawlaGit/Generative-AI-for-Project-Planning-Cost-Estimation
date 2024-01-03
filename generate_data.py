import openai
import pandas as pd
import re
import psycopg2
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time

api_key = "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"

# Initialize the OpenAI client
openai.api_key = api_key

# Connection parameters for PostgreSQL
db_host = "127.0.0.1"
db_port = "5432"
db_name = "CostEstimates"
db_user = "postgres"
db_password = "1234"


# SQLAlchemy connection string
connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create an SQLAlchemy engine
engine = create_engine(connection_string)

db_queries = Counter('db_queries', 'Number of database queries')
api_calls = Counter('api_calls', 'Number of API calls made')
errors = Counter('errors', 'Number of errors encountered')
response_time = Histogram('response_time', 'Response time for API calls')

start_http_server(8000)

# Load the test data from PostgreSQL table
try:
    test_data = pd.read_sql('SELECT * FROM "TestingData"', engine)
    db_queries.inc()  # Increment the database query counter
    print(test_data.columns)
except Exception as e:
    errors.inc()  # Increment the error counter
    print(f"Error: Unable to fetch data from the database. Detailed error: {e}")




# Create an empty list to store effort, duration, coding, and testing estimates
object_points_list = []
estimated_effort_list = []




def extract_numeric_value(response_text):
    match = re.search(r'\d+(\.\d+)?', response_text)
    return float(match.group()) if match else None

for index, row in test_data.iterrows():

    object_points_prompt = f"Calculate Object Points as follows:\n"
    object_points_prompt += f"1. Sum the Number of screens ({row['numberofscreens']}).\n"
    object_points_prompt += f"2. Sum the Number of reports ({row['numberofreports']}).\n"
    object_points_prompt += f"The final Object Points are the sum of these values.\n"

    try:
        with response_time.time():
            response_object_points = openai.Completion.create(
                engine="text-davinci-003",
                prompt=object_points_prompt,
                temperature=0.7,
                max_tokens=50,
                n=1,
                stop=None
            )
        api_calls.inc()
    except Exception as e:
        errors.inc()  # Increment the error counter
        print(f"Error: Unable to make an API call. Detailed error: {e}")


    
    # Prepare the prompt for the first part of Estimated Effort (Duration * Team members)
    effort_part1_prompt = f"Calculate the first part of Estimated Effort using the formula: Estimated duration ({row['estimatedduration']} days) * Dedicated Team members ({row['dedicatedteammembers']})."

    # Make an API call for the first part of Estimated Effort
    try:
        with response_time.time():
            response_part1 = openai.Completion.create(
                engine="text-davinci-003",
                prompt=effort_part1_prompt,
                temperature=0.7,
                max_tokens=50,
                n=1,
                stop=None
            )
        api_calls.inc()  # Increment the API call counter
    except Exception as e:
        errors.inc()  # Increment the error counter
        print(f"Error: Unable to make an API call. Detailed error: {e}")


    # Extract the first part of Estimated Effort
    effort_part1 = extract_numeric_value(response_part1.choices[0].text) if response_part1.choices else None

    # Calculate the second part of Estimated Effort (Remaining Team members * 0.5)
    remaining_team_members = row['teamsize'] - row['dedicatedteammembers']
    effort_part2_prompt = f"Calculate the second part of Estimated Effort using the formula: Remaining Team members ({remaining_team_members}) * 0.5."

    # Make an API call for the second part of Estimated Effort
    try:
        with response_time.time():
            response_part2 = openai.Completion.create(
                engine="text-davinci-003",
                prompt=effort_part2_prompt,
                temperature=0.7,
                max_tokens=50,
                n=1,
                stop=None
            )
        api_calls.inc()  # Increment the API call counter

    except Exception as e:
        errors.inc()  # Increment the error counter
        print(f"Error: Unable to make an API call. Detailed error: {e}")


    # Extract the second part of Estimated Effort
    effort_part2 = extract_numeric_value(response_part2.choices[0].text) if response_part2.choices else None

    # Calculate the final Estimated Effort in Python
    final_estimated_effort = (effort_part1 + effort_part2) * (row['dailyworkinghours'] * 22)

    # Append Object Points and Estimated Effort to respective lists
    object_points_list.append(extract_numeric_value(response_object_points.choices[0].text) if response_object_points.choices else None)
    estimated_effort_list.append(final_estimated_effort)



# Add Object Points and Estimated Effort to the test_data
test_data['objectpointspredicted'] = object_points_list
test_data['estimatedeffortpredicted'] = estimated_effort_list



try:
    test_data.to_sql('TestingData', engine, if_exists='replace', index=False)
    db_queries.inc()  # Increment the database query counter (considered as a write operation here)
    print("Predicted Estimated Effort and Object Points estimates added and saved to the PostgreSQL database.")
except Exception as e:
    errors.inc()  # Increment the general error counter for any kind of errors
    print(f"Error: Unable to save data to the database. Detailed error: {e}")



# Check and handle NaN or infinite values in 'actualobjectpoints' and 'objectpointspredicted'
test_data['actualobjectpoints'] = test_data['actualobjectpoints'].replace([np.inf, -np.inf], np.nan)
test_data['objectpointspredicted'] = test_data['objectpointspredicted'].replace([np.inf, -np.inf], np.nan)

# Option 1: Fill NaN values with a default number (e.g., 0)
test_data.fillna(0, inplace=True)

# Define a function to make the API call and execute the SQL query
def generate_and_execute_query(prompt):
    try:
        with response_time.time():
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=50,
                n=1,
                stop=None
            )
        api_calls.inc()  # Increment the API call counter

    except Exception as e:
        errors.inc()  # Increment the error counter
        print(f"Error: Unable to make an API call. Detailed error: {e}")

    
    generated_query = response.choices[0].text.strip() if response.choices else None

    if generated_query:
        print(f"Generated SQL Query: {generated_query}")
        try:
            result_data = pd.read_sql_query(generated_query, engine)
            db_queries.inc()  # Increment the database query counter
            print("Result:")
            print(result_data)
        except Exception as e:
            errors.inc()  # Increment the error counter
            print(f"Error: Unable to execute the generated SQL query. Detailed error: {e}")
    else:
        errors.inc()  # Increment the error counter for failing to generate a query
        print("Error: Unable to generate a valid SQL query.")

# Example prompts for the OpenAI API call
prompts = [
    # Your initial prompt
    "Create an SQL query suitable for a PostgreSQL database to retrieve the first 10 rows from a table named 'TestingData'. Ensure that the table name is enclosed in double quotes to adhere to PostgreSQL's case sensitivity. The query should select all columns from these rows.",
    # Additional prompts for max and min values
    "Create an SQL query suitable for a PostgreSQL database to retrieve the average value of the 'objectpointspredicted' column from a table named 'TestingData'. Ensure that the table name is enclosed in double quotes to adhere to PostgreSQL's case sensitivity.",
    "Create an SQL query suitable for a PostgreSQL database to retrieve the average value of the 'estimatedeffortpredicted' column from a table named 'TestingData'. Ensure that the table name is enclosed in double quotes to adhere to PostgreSQL's case sensitivity."

]

# Execute the function for each prompt
for prompt in prompts:
    generate_and_execute_query(prompt)



# Plotting 'actualestimatedeffort' vs 'estimatedeffortpredicted' with enhancements
plt.figure(figsize=(12, 8))
plt.scatter(test_data['actualestimatedeffort'], test_data['estimatedeffortpredicted'], alpha=0.6, edgecolors='w', s=80)
z = np.polyfit(test_data['actualestimatedeffort'], test_data['estimatedeffortpredicted'], 1)
p = np.poly1d(z)
plt.plot(test_data['actualestimatedeffort'], p(test_data['actualestimatedeffort']), "r--")
plt.title('Actual vs. Predicted Estimated Effort', fontsize=16)
plt.xlabel('Actual Estimated Effort', fontsize=14)
plt.ylabel('Predicted Estimated Effort', fontsize=14)
plt.grid(True)
plt.tight_layout()

# Show the first plot
plt.show()

# Plotting 'actualobjectpoints' vs 'objectpointspredicted' with enhancements
plt.figure(figsize=(12, 8))
plt.scatter(test_data['actualobjectpoints'], test_data['objectpointspredicted'], alpha=0.6, edgecolors='w', s=80, color='green')
z = np.polyfit(test_data['actualobjectpoints'], test_data['objectpointspredicted'], 1)
p = np.poly1d(z)
plt.plot(test_data['actualobjectpoints'], p(test_data['actualobjectpoints']), "r--")
plt.title('Actual vs. Predicted Object Points', fontsize=16)
plt.xlabel('Actual Object Points', fontsize=14)
plt.ylabel('Predicted Object Points', fontsize=14)
plt.grid(True)
plt.tight_layout()

# Show the second plot
plt.show()

# Option 2: Drop rows with NaN values
#test_data.dropna(subset=['actualobjectpoints', 'objectpointspredicted'], inplace=True)

r2_object_points = r2_score(test_data['actualobjectpoints'], test_data['objectpointspredicted'])

print(f"Object Points - R² Score: {r2_object_points}")


r2_effort = r2_score(test_data['actualestimatedeffort'], test_data['estimatedeffortpredicted'])

print(f"Estimated Effort - R² Score: {r2_effort}")

time.sleep(1000)