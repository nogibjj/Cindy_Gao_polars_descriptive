[![Assignment 1 Python Application Test with Github Actions from Cindy](https://github.com/nogibjj/Cindy_Gao_polars_descriptive/actions/workflows/actions.yml/badge.svg)](https://github.com/nogibjj/Cindy_Gao_polars_descriptive/actions/workflows/actions.yml)
# Cindy_Gao_polars_descriptive

- .devcontainer: It sets up a consistent development environment across different machines.
- .github/workflows: It defines automated workflows for CI/CD tasks
- Makefile: It manages tasks like installing dependencies, formatting code, linting, and testing- requirements.txt
- requirements: Lists the Python packages required by the project
- main.py: Contains the main code and functions for the project
- Boxplot.png: A data visualization of boxplot for variable Y
- Boxplot.md: A markdown for the boxplot image, and the mean, median and standard deviation calculation
- test_main.py: Contains test cases for main.py
- summary_test_report.pdf: A pdf containing summary of the results of the test functions and the boxplot image.
- README.md: Provides documentation and their information for the project


# Purpose for this project:
1. Use Polars to read csv file
2. Write polars functions for calculating mean, median and standard deviation for variables
3. Compare the functions for the accuracy
4. Create data visualization (a boxplot) for variable Y
5. Create a markdown or pdf file for the summary
6. Use Profiler to compare between the execution time Pandas and Polars


# Url to the raw data:
data = "https://raw.githubusercontent.com/anlane611/datasets/main/population.csv"


# Statistic Summary for the data:
<img width="409" alt="image" src="https://github.com/user-attachments/assets/bf7101be-b4f6-486f-a57c-599e940122c0">





# Below is the boxplot for variable Y:
![Boxplot for Variable Y](boxplot.png)






