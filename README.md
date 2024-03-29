# The use of generative AI to generate estimates for the project plan, design document, source code, test plan and test cases, and user API documentation
## Problem Statement
The purpose of this project is to use generative AI capabilities (GPT-3 from OpenAI to improve the
automation of project management tasks. It includes fully integrating a PostgreSQL database and
functioning as a repository for project data that we got from the SEERA Software Cost Estimation
Dataset.  
Our code interacts with the PostgreSQL database, effectively fetching and generating project estimates.
Additionally, OpenAI's intelligent virtual assistant is an essential part, engaged not only for data retrieval
but also for content generation, responding to prompts for project plans, design documents, and other
related artifacts.  
In addition to its generative role, the virtual assistant is important in calculating estimations for effort
duration, using the formula established within the SEERA dataset as a standard. This formula is essential
to our approach, giving a systematic means to calculate estimated durations and store them within the
PostgreSQL database.  
We determine the accuracy and performance of our estimations through a comparison between the
estimated durations and the actual durations derived from the SEERA dataset. This analysis serves as an
essential evaluation method, confirming the reliability and applicability of our generative AI model.
Our project seeks to go beyond the standard models of project management, providing a fusion of
generative AI and accurate performance validation.

## Results
### R² Score
We implemented the coefficient of determination (R²) to evaluate our model's ability to predict object
points and estimated effort accurately. The score provides insights into the goodness of fit between
predicted and actual values. A higher R² indicates a better model fit.

The R² scores achieved in our analysis reflect a high level of accuracy in our model predictions.  
**Object Points - R² Score: 0.940763681596664**    
**Estimated Effort - R² Score: 0.9697761036280876**   

### Our code generates scatter plots, which visualize the relationship between actual and predicted values for both object points and estimated effort, providing an assessment of our model's performance

![image](https://github.com/BhavyaChawlaGit/Generative-AI-for-Cost-Estimation-in-Project-Planning/assets/112718303/7507bacd-a204-4cc3-b88e-12a580268420)


![image](https://github.com/BhavyaChawlaGit/Generative-AI-for-Cost-Estimation-in-Project-Planning/assets/112718303/08b14776-9ec3-4e4e-8530-1efe843c50c7)

## Monitoring with Prometheus
It ensure the reliability of our system, we have integrated Prometheus for monitoring OpenAI GPT-3 API
calls, as also taught in the course. Prometheus enabled real-time tracking of key metrics, such as response
time, error counts, API calls, and the count of SQL queries.

![image](https://github.com/BhavyaChawlaGit/Generative-AI-for-Cost-Estimation-in-Project-Planning/assets/112718303/0e86dfd4-64b6-44e7-a739-9268703f5eab)

![image](https://github.com/BhavyaChawlaGit/Generative-AI-for-Cost-Estimation-in-Project-Planning/assets/112718303/7843777f-2ac0-48d0-9749-965aec067907)











