# 19-th International Fryderyk Chopin Piano Competition 2025 Analysis 

**Disclaimer:** This is an unofficial analysis created for private, hobbyist purposes. It is based on scoring data made publicly available by the Fryderyk Chopin Institute.

This repository contains a Python script to analyze the Chopin Competition 2025 scores and generate visualizations.

## Generated Output

- [Console Output](output.txt)

### Juror Score Histograms

#### Without Correction

- [Stage 1 Histograms](histograms/stage_1_histograms.png)
- [Stage 2 Histograms](histograms/stage_2_histograms.png)
- [Stage 3 Histograms](histograms/stage_3_histograms.png)
- [Final Stage Histograms](histograms/stage_final_histograms.png)
- [All Stages Histograms](histograms/stage_ALL_histograms.png)

#### With Correction

- [Stage 1 Histograms](histograms/stage_1_histograms_korekta.png)
- [Stage 2 Histograms](histograms/stage_2_histograms_korekta.png)
- [Stage 3 Histograms](histograms/stage_3_histograms_korekta.png)
- [Final Stage Histograms](histograms/stage_final_histograms_korekta.png)
- [All Stages Histograms](histograms/stage_ALL_histograms_korekta.png)

### [Jurors correlations](https://marcinprzeradzki.github.io/chopin-19/corr_punkty.html)


## How to Run

For those not familiar with Python, here are the steps to run the analysis on your own computer:

1.  **Prerequisites:** Ensure you have the following tools installed on your system:
    -   Python (version 3.8 or higher recommended)
    -   Git

2.  **Clone the Repository:**
    - Open a terminal (on Windows, you can use Command Prompt or PowerShell; on macOS or Linux, use Terminal).
    - Navigate to the directory where you want to store the project.
    - Run the following command to clone the repository:
      ```bash
      git clone https://github.com/marcinprzeradzki/chopin-19.git
      cd chopin-19
      ```

3.  **Set up a Virtual Environment:**
    - Create a virtual environment to manage the project's dependencies. Run the following command in the project's root directory:
      ```bash
      python -m venv .venv
      ```

4.  **Activate the Virtual Environment:**
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```bash
      source .venv/bin/activate
      ```

5.  **Install Dependencies:**
    - With the virtual environment activated, install the required Python libraries by running:
      ```bash
      pip install -r requirements.txt
      ```

6.  **Run the Analysis:**
    - Now you can run the analysis script:
      ```bash
      python analyzer.py
      ```

7.  **View the Output:**
    - The script will generate an `output.txt` file with the analysis results and a `histograms` directory with PNG images of the histograms.
