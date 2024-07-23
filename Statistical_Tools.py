import os
import pyodbc
import pandas as pd
import json
import os
from datetime import timedelta

import requests
from langchain.tools import tool

import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools

os.environ["OPENAI_API_KEY"] = "YOUR_APIKEY"

import json  # Import the JSON module to parse JSON strings
from langchain_core.agents import AgentFinish

agent_finishes  = []

import json
from typing import Union, List, Tuple, Dict
from langchain.schema import AgentFinish

call_number = 0



############################################################################################################################################################################################
#####Logging Agent
############################################################################################################################################################################################

def print_agent_output(agent_output: Union[str, List[Tuple[Dict, str]], AgentFinish], agent_name: str = 'Generic call'):
    global call_number  # Declare call_number as a global variable
    call_number += 1
    with open("crew_callback_logs.txt", "a") as log_file:
        # Try to parse the output if it is a JSON string
        if isinstance(agent_output, str):
            try:
                agent_output = json.loads(agent_output)  # Attempt to parse the JSON string
            except json.JSONDecodeError:
                pass  # If there's an error, leave agent_output as is

        # Check if the output is a list of tuples as in the first case
        if isinstance(agent_output, list) and all(isinstance(item, tuple) for item in agent_output):
            print(f"-{call_number}----Dict------------------------------------------", file=log_file)
            for action, description in agent_output:
                # Print attributes based on assumed structure
                print(f"Agent Name: {agent_name}", file=log_file)
                print(f"Tool used: {getattr(action, 'tool', 'Unknown')}", file=log_file)
                print(f"Tool input: {getattr(action, 'tool_input', 'Unknown')}", file=log_file)
                print(f"Action log: {getattr(action, 'log', 'Unknown')}", file=log_file)
                print(f"Description: {description}", file=log_file)
                print("--------------------------------------------------", file=log_file)

        # Check if the output is a dictionary as in the second case
        elif isinstance(agent_output, AgentFinish):
            print(f"-{call_number}----AgentFinish---------------------------------------", file=log_file)
            print(f"Agent Name: {agent_name}", file=log_file)
            agent_finishes.append(agent_output)
            # Extracting 'output' and 'log' from the nested 'return_values' if they exist
            output = agent_output.return_values
            # log = agent_output.get('log', 'No log available')
            print(f"AgentFinish Output: {output['output']}", file=log_file)
            # print(f"Log: {log}", file=log_file)
            # print(f"AgentFinish: {agent_output}", file=log_file)
            print("--------------------------------------------------", file=log_file)

        # Handle unexpected formats
        else:
            # If the format is unknown, print out the input directly
            print(f"-{call_number}-Unknown format of agent_output:", file=log_file)
            print(type(agent_output), file=log_file)
            print(agent_output, file=log_file)

from crewai import Crew, Agent, Task, Process
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_openai import ChatOpenAI

# Initialize the OpenAI GPT-4 language model
OpenAIGPT4TURBO = ChatOpenAI(
    model="gpt-4-turbo-preview"
)

from datetime import datetime
from random import randint
from langchain.tools import tool
import pyodbc

######################################################################################################################################################
#####Custom Tools
######################################################################################################################################################

@tool("save_content")
def save_content(task_output):
    """Useful to save content to a markdown file"""
    print('in the save markdown tool')
    # Get today's date in the format YYYY-MM-DD
    today_date = datetime.now().strftime('%Y-%m-%d')
    # Set the filename with today's date
    filename = f"{today_date}_{randint(0,100)}.md"
    # Write the task output to the markdown file
    with open(filename, 'w') as file:
        file.write(task_output)
        # file.write(task_output.result)

    print(f"Blog post saved as {filename}")

    return f"Blog post saved as {filename}, please tell the user we are finished"

@tool("centrality_statistics")
def centrality_statisitcs(data):
    """Useful for calcualting centrality statistics on a given array of numbers"""
    from typing import List, Optional, Union

    def calculate_average(data: List[float]) -> Optional[float]:
        if not data:
            return None
        average = sum(data) / len(data)
        return round(average, 2)

    def calculate_median(data: List[float]) -> Optional[float]:
        if not data:
            return None
        sorted_data = sorted(data)
        n = len(sorted_data)
        middle = n // 2
        if n % 2 == 0:
            median = (sorted_data[middle - 1] + sorted_data[middle]) / 2
        else:
            median = sorted_data[middle]
        return round(median, 2)

    try:
        numerical_data = []
        if isinstance(data, str):
            numerical_data = [float(value.strip()) for value in data.split(',') if value.strip()]
        elif isinstance(data, list):
            numerical_data = data

        if not numerical_data:
            return "Dataset is empty."

        result = {
            'Average': calculate_average(numerical_data),
            'Median': calculate_median(numerical_data)
        }
        return result

    except ValueError:
        return "Invalid input. Please provide a list of numbers or a valid string representation."

@tool("variance_statistics")
def variance_statistics(data):
    """Useful for calcualting variance statistics on a given array of numbers"""
    from typing import List, Union, Optional
    import math
    import numpy as np

    def calculate_variance(data: List[Union[int, float]]) -> Optional[float]:
        if len(data) < 2:
            return None
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return round(variance, 2)

    def calculate_standard_deviation(data: List[Union[int, float]]) -> Optional[float]:
        variance = calculate_variance(data)
        if variance is not None:
            standard_deviation = round(math.sqrt(variance), 2)
            return standard_deviation
        return None

    def calculate_interquartile_range(data: List[Union[int, float]]) -> Optional[float]:
        if not data:
            return None
        q3, q1 = np.percentile(data, [75 ,25])
        iqr = q3 - q1
        return round(iqr, 2)

    try:
        numerical_data = []
        if isinstance(data, str):
            numerical_data = [float(value.strip()) for value in data.split(',') if value.strip()]
        elif isinstance(data, list):
            numerical_data = data

        if not numerical_data:
            return "Dataset is empty."

        result = {
            'Variance': calculate_variance(numerical_data),
            'Standard Deviation': calculate_standard_deviation(numerical_data),
            'Interquartile Range': calculate_interquartile_range(numerical_data)
        }
        return result

    except ValueError:
        return "Invalid input. Please provide a list of numbers or a valid string representation."

############################################################################################################################################################################################
#####SQL Tools
############################################################################################################################################################################################


@tool("sql_query")
def sql_query(start_date, end_date):
    """Executes and pulls down raw reviews in a given time period"""
    # Connection string
    connection_string = (YOUR_SQL_CONNECTION_STRING
    )

    # Connect to the database
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    
    # Execute the stored procedure and fetch all results
    cursor.execute("EXEC [dbo].[CEA_Reviews] @StartDate = ?, @EndDate = ?",
                   (start_date, end_date))
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    
    # Convert results to DataFrame
    df = pd.DataFrame.from_records(results, columns=columns)
    

    # Calculate statistics
    total_reviews = len(df)
    count_controllable = len(df[df['exceptions'] == 'controllable'])
    count_uncontrollable = len(df[df['exceptions'] == 'uncontrollable'])
    percent_controllable = (count_controllable / total_reviews) * 100 if total_reviews > 0 else 0
    percent_uncontrollable = (count_uncontrollable / total_reviews) * 100 if total_reviews > 0 else 0

    # Creating stats DataFrame
    stats = pd.DataFrame({
        "Total Reviews": [total_reviews],
        "Number of Controllable Exceptions": [count_controllable],
        "Number of Uncontrollable Exceptions": [count_uncontrollable],
        "% Controllable": [percent_controllable],
        "% Uncontrollable": [percent_uncontrollable]
    })
    
    # Clean up
    cursor.close()
    conn.close()
    
    # Print stats DataFrame
    print(stats)
    
    return stats

@tool("sql_time_period_query")
def sql_time_period_query():
    """Generates stats on controllable and uncontrollable reviews for detailed time periods"""
    # Current date
    today = datetime.now()
    first_day_of_year = datetime(today.year, 1, 1)
    first_day_of_last_year = datetime(today.year - 1, 1, 1)
    same_day_last_year = datetime(today.year - 1, today.month, today.day)

    # Connection string
    connection_string = (
Your_Connection_String
    )

    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    # Time periods dictionary
    periods = {
        "YTD": (first_day_of_year, today),
        "Last Year Same Period": (first_day_of_last_year, same_day_last_year),
    }

    # Adding month-by-month for the last 12 months
    for i in range(12):
        end_date = datetime(today.year, today.month, 1) - timedelta(days=i*31)
        start_date = datetime(end_date.year, end_date.month, 1)
        period_name = f"{start_date.strftime('%B %Y')}"
        periods[period_name] = (start_date, end_date)

    results = {}

    for period, dates in periods.items():
        cursor.execute("EXEC [dbo].[CEA_Reviews] @StartDate = ?, @EndDate = ?",
                       dates)
        columns = [column[0] for column in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame.from_records(data, columns=columns)
        total_reviews = len(df)
        count_controllable = len(df[df['exceptions'] == 'controllable'])
        count_uncontrollable = len(df[df['exceptions'] == 'uncontrollable'])
        percent_controllable = (count_controllable / total_reviews) * 100 if total_reviews > 0 else 0
        percent_uncontrollable = (count_uncontrollable / total_reviews) * 100 if total_reviews > 0 else 0
        
        results[period] = {
            "Total Reviews": total_reviews,
            "Number of Controllable Exceptions": count_controllable,
            "Number of Uncontrollable Exceptions": count_uncontrollable,
            "% Controllable": percent_controllable,
            "% Uncontrollable": percent_uncontrollable
        }

    # Convert results dictionary to DataFrame
    stats_df = pd.DataFrame(results).T  # Transpose to swap rows and columns

    cursor.close()
    conn.close()
    
    print(stats_df)
    return stats_df

search_tool = DuckDuckGoSearchRun()

# Loading Human Tools
human_tools = load_tools(["human"])
from crewai import Agent



######################################################################################################################################################
#####Agents
######################################################################################################################################################

# Define your agents with roles and goals
analytics_provider = Agent(
    role='Review executor',
    goal='execute SQL query',
    backstory="""Your goal is to respond to the question the user asks about reviews.""",
    verbose=True,
    allow_delegation=False,  # Ensure this attribute is set correctly
    llm=OpenAIGPT4TURBO,
    max_iter=5,
    memory=True,
    step_callback=lambda x: print_agent_output(x,"Senior Research Analyst Agent"),
    tools=[sql_query, sql_time_period_query] + human_tools  # Ensure tools are assigned correctly
)

archiver = Agent(
    role='File Archiver',
    goal='Take in information and write it to a Markdown file',
    backstory="""You are a efficient and simple agent that gets data and saves it to a markdown file. in a quick and efficient manner""",
    llm=OpenAIGPT4TURBO,
    # allow_delegation=False,
    verbose=True,
    step_callback=lambda x: print_agent_output(x,"Archiver Agent"),
    tools=[save_content],
)

from datetime import datetime




######################################################################################################################################################
#####Tasks
######################################################################################################################################################

review_conttrollable_uncontrollable = Task(
    description="""Using SQL tools, pull down reviews from the database using the parameters specificed by the user.
                    If the user does not specify a timeframe, use the human tool to ask. Based on the dataframe
                    that is generated by the SQL tool, take all the reviews and provide a summary on it.""",
    expected_output="""A summary of the reviews that were pulled down from the database""",
    Agent=analytics_provider
)

time_period_narative = Task(
  description="""Execute tools that provide an statistical analysis on reviews that will help give insight on what
  the hotel should focus on. 
""",
  expected_output="""Using past statistics to describe the current sitiuation and trends of past data 
  of controallble and uncontraollable reviews.""",
  agent=analytics_provider
)

raw_reviews = Task(
  description="""Using the human tool, ask the user what they are wanting. You will execute a tool that 
  will pull down data in certain time period and anaylze that datafraem to determine what the course of action 
  will be
""",
  expected_output="""A compleling, factual response that is easy to understand and digestible for a hotel on what
  he should focus on.""",
  agent=analytics_provider
)

review_analysis= Task(
    description=
    """Using sql_query tool to pull down reviews by using the human tool to ask the user what time dimension, or by using the 
    human tool to ask a timeframe, take maximum 20 reviews from the sql query and provide an action plan on 
    what the hotel should do to improve their reviews. But this should be in an action plan and using statistical
    reasoning from another task. Most importantly, do not give general suggestions, but concrete action
    itens citing actual reviews people have put in that specifc time period. Only use the human tool to ask for time dimension
    which will be used for the sql_query tool. Again, this is the most important task, use the acutal
    reviews created from the SQL_query tool to create an action plan, but do this using actual 
    examples from the reviews. Do not have this general, but specific to the reviews.""",
    expected_output="""A detailed action plan on what the hotel should do to improve their reviews.""",
    agent=analytics_provider
)

review_language = Task(
    description="""Using the output from the review_conttrollable_uncontrollable task, take the output and dress
    up the language to provide an action plan for the hotel. This should be a response that is easy to understand with no
    redundancies. You must use actual reviews to state the action plan.""",
    expected_output="""A detailed action plan on what the hotel should do to improve their reviews.""",
    Agent=analytics_provider
)
# write_the_content

saving_the_output = Task(
  description="""Taking the post created by the write the content, take this and save it to a markdown file.
  Your final answer MUST be a response must be showing that the file was saved .""",
  expected_output='A saved file name',
  agent=archiver
)



######################################################################################################################################################
#####Initiate the Crew
######################################################################################################################################################

from crewai import Crew, Process

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[analytics_provider, archiver],
    tasks=[review_conttrollable_uncontrollable, review_language, saving_the_output],
    verbose=2,
    process=Process.sequential,
    full_output=True,
    share_crew=False,
    step_callback=lambda x: print_agent_output(x,"MasterCrew Agent")
)

# Kick off the crew's work
results = crew.kickoff()

# Print the results
print("Crew Work Results:")
print(results)
