<h1 align="center">Brain Stroke Analysis & Prediction</h1>

<h3 align="center">A Streamlit project presentation for exploratory analysis and neural-network-based stroke prediction</h3>

<p align="center">
  Explore the dataset step by step, review the analysis workflow, and prepare patient inputs for interactive model testing.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?style=flat&logo=python&logoColor=white" alt="Python 3.14" />
  <img src="https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit 1.58.0" />
  <img src="https://img.shields.io/badge/Pandas-3.0.5-150458?style=flat&logo=pandas&logoColor=white" alt="Pandas 3.0.5" />
  <img src="https://img.shields.io/badge/Seaborn-0.13.2-4C9A9A?style=flat" alt="Seaborn 0.13.2" />
  <img src="https://img.shields.io/badge/Scikit--learn-1.9.1-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit-learn 1.9.1" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat" alt="PRs welcome" />
</p>

---

## Overview

**Brain Stroke Analysis & Prediction** is a **Streamlit data-science presentation app** that explains a healthcare stroke dataset, walks through cleaning and exploratory analysis, and prepares an interactive interface for testing a neural-network model. It helps portfolio reviewers and visitors understand both the analytical reasoning and the intended prediction workflow.

The app loads the healthcare stroke dataset, removes duplicate and incomplete records, presents teal-themed visualizations, and organizes the work into overview, dataset, exploration, method, report, and model-testing views. The exported neural-network model and its matching preprocessing artifacts will be connected when they are added to the project folder.

## Demo and Screenshots

Run the app locally to view the current presentation:

```text
http://localhost:8501
```

No screenshot assets or hosted demo are currently included in the repository.

## Features

- Project overview explaining the stroke-analysis motivation.
- Dataset preview and documented cleaning decisions.
- Exploratory visualizations for outcome balance, age, glucose level, and correlations.
- Step-by-step methodology covering loading, cleaning, transformation, splitting, scaling, and evaluation.
- Interactive patient-parameter form prepared for neural-network inference.
- Consistent teal visualization theme matching the original notebook analysis.
- Cached dataset loading for a responsive Streamlit experience.

## Tech Stack

| Concern | Technology | Purpose |
| --- | --- | --- |
| Language | Python 3.14 | Application and analysis logic |
| App framework | Streamlit 1.58.0 | Interactive presentation and input workflow |
| Data processing | Pandas 3.0.5 | Loading, cleaning, and transforming tabular data |
| Numerical computing | NumPy | Numerical operations used by the analysis stack |
| Visualization | Matplotlib 3.11.2 and Seaborn 0.13.2 | Analysis charts and correlation maps |
| Machine learning utilities | Scikit-learn 1.9.1 | Planned preprocessing and evaluation integration |

## Project Structure

```text
Brain_stroke_analysis/
├── app.py                                      # Streamlit presentation application
├── brain_stroke_analysis2.py                   # Exported notebook analysis script
├── brain_stroke_analysis2.ipynb                 # Original Jupyter notebook
├── brain_stroke_dataset/
│   └── healthcare-dataset-stroke-data.csv       # Local analysis dataset
├── brain_stroke_dataset.zip                     # Compressed dataset archive
├── requirements.txt                             # Python dependencies for the app
└── README.md                                    # Project documentation
```

The current data flow is:

```text
CSV dataset -> cached loading -> cleaning and exploratory views -> prediction form
```

The neural-network model file and preprocessing artifacts will be added to this structure after export from Colab.

## Quick Start

### Prerequisites

- Python 3.10 or newer.
- The dataset at `brain_stroke_dataset/healthcare-dataset-stroke-data.csv`.
- A terminal opened in the repository directory.

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

On this development machine, Streamlit is installed in Anaconda. The equivalent launch command is:

```powershell
C:\Users\tebio\anaconda3\python.exe -m pip install -r requirements.txt
```

### Launch the application

```powershell
C:\Users\tebio\anaconda3\python.exe -m streamlit run app.py --server.port 8501
```

Then open [http://localhost:8501](http://localhost:8501) in a browser. A successful launch displays the app navigation in the Streamlit page.

## How It Works

1. **Load:** Read the CSV dataset from the repository-local dataset directory.
2. **Clean:** Remove duplicate records, drop rows with missing BMI, and exclude the `Other` gender category used by the original analysis.
3. **Explore:** Present outcome balance, age and glucose patterns, and numeric correlations using the project palette.
4. **Explain:** Show the intended transformation, train/test split, scaling, and evaluation stages.
5. **Infer:** Collect patient parameters through a form and pass the same feature representation to the exported neural network once its artifacts are available.

The final inference path must use the same preprocessing decisions and feature ordering as training. The model should not receive raw form values until categorical encoding and numeric scaling have been reproduced exactly.

## Results and Limitations

The current app demonstrates the presentation workflow and exploratory analysis. It does not yet report verified neural-network metrics or generate live predictions because the exported model and matching preprocessing artifacts have not been added.

Known limitations:

- The current Streamlit form is an integration scaffold and displays submitted inputs until the model is connected.
- No accuracy, precision, recall, F1-score, or confusion-matrix values are claimed here without the exported model evaluation evidence.
- Stroke datasets are commonly imbalanced; accuracy alone will not be sufficient for assessing the final model.
- This project is educational and must not be used as a medical diagnosis or clinical decision system.

## Usage

Use the sidebar to move through the presentation:

| View | Purpose |
| --- | --- |
| Overview | Introduces the project and its motivation |
| Dataset | Shows sample records and cleaning decisions |
| Exploration | Displays the main exploratory charts |
| Method | Explains the analytical workflow |
| Test the model | Collects patient parameters for future inference |

## Configuration

The app currently uses repository-local paths and does not require environment variables or secrets.

| Setting | Current value |
| --- | --- |
| Dataset path | `brain_stroke_dataset/healthcare-dataset-stroke-data.csv` |
| Default Streamlit port | `8501` in the documented launch command |
| Visualization background | `#E0F7FA` |
| Primary visualization accent | `#00796B` |

When the neural-network model is added, document its filename, expected input shape, preprocessing artifact, and any required configuration here.

## Development and Deployment

Run a syntax check before starting the app:

```powershell
python -m py_compile app.py
```

Run the local app with:

```powershell
C:\Users\tebio\anaconda3\python.exe -m streamlit run app.py --server.port 8501
```

The application is currently prepared for local Streamlit use. Deployment has not been configured or verified for Streamlit Community Cloud, Docker, or another hosting provider.

## Roadmap

- [ ] Add the exported neural-network model to the project folder.
- [ ] Add the exact training-time preprocessing artifact or reproduce it in a shared preprocessing module.
- [ ] Connect the prediction form to neural-network inference.
- [ ] Display prediction probability and a clearly worded result state.
- [ ] Add verified evaluation metrics, confusion matrix, and training-history report.
- [ ] Add screenshots of the completed application.
- [ ] Verify deployment on a selected hosting platform.

## Contributing, License, and Acknowledgements

This is a portfolio project. Improvements to the presentation, reproducibility, documentation, and model integration are welcome through pull requests or issue discussions.

No license file is currently included in the repository, so no open-source license is claimed.

The project uses the healthcare stroke dataset stored in `brain_stroke_dataset/healthcare-dataset-stroke-data.csv`, along with Python, Pandas, Matplotlib, Seaborn, Scikit-learn, and Streamlit.
