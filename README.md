# Online Preference-Based Reward Learning: Optimal Query Timing

This repository provides the implementation and supporting materials for the project "Online Preference-Based Reward Learning in the Presence of Human Irrationalities and Reaction Delay," as detailed in the CMPUT 656 course report by Masoud Jafaripour.

## Table of Contents

* [Introduction](#introduction)
* [Key Contributions](#key-contributions)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Usage](#usage)

  * [Training the Pretrained Model](#training-the-pretrained-model)
  * [Query Timing Algorithm](#query-timing-algorithm)
  * [Human Reaction Time Fitting](#human-reaction-time-fitting)
* [Experiments](#experiments)

  * [Human User Study](#human-user-study)
  * [Reaction Time Models Comparison](#reaction-time-models-comparison)
* [Results and Artifacts](#results-and-artifacts)
* [Future Work](#future-work)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Introduction

Designing reward functions for complex tasks in reinforcement learning is often impractical and time-consuming. Preference-based Reinforcement Learning (PbRL) offers an alternative by inferring rewards from human feedback. However, the timing of queries is critical: human irrationalities and non-instantaneous responses can degrade learning performance if not properly accounted for citeturn0file0.

This project proposes an algorithm to determine the **optimal query timing** by predicting the Expected Value of Information (EVOI) over future timesteps and incorporating a probabilistic model of human reaction delays. By combining these elements, the agent maximizes informative feedback while compensating for human non-idealities.

## Key Contributions

1. **EVOI Prediction Scheme**: A method to estimate EVOI across a future horizon and select the optimal local maximum within the querying span.
2. **Human Reaction Time Model**: Statistical modeling of human response delays using Gamma (and Gaussian) distributions fitted from user studies and literature data.
3. **Irrationality Incorporation**: Extension of the stochastic human preference model to capture biases such as myopia, skipping, and equal preferences.
4. **Unified Query Design Algorithm**: Integration of EVOI prediction and reaction time modeling in a single algorithm to determine when to pose preference queries in an online fashion.

## Repository Structure

```
├── data/                           # Raw and processed data files
│   ├── reaction_times/             # User study recordings and literature models
│   └── environments/               # Simulation scenarios (e.g., highway-env configs)
├── src/                            # Core source code
│   ├── agent/                      # Pretrained model and Q-function implementations
│   ├── query_timing/               # EVOI prediction and query scheduling modules
│   ├── human_model/                # Reaction time fitting and irrationality simulators
│   ├── experiments/                # Experiment scripts and evaluation routines
│   ├── utils/                      # Common utilities and helpers
│   └── api/                        # RESTful API for live querying (experimental)
├── notebooks/                      # Jupyter notebooks for analysis and visualization
├── results/                        # Generated plots, EVOI curves, and logs
├── videos/                         # Demonstration videos of user studies and algorithms
├── docs/                           # Supplementary documentation and figures
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup script
└── README.md                       # This file
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<username>/optimal-query-timing.git
   cd optimal-query-timing
   ```
2. Create and activate a Python environment (Python 3.8+):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Pretrained Model

Pretrain or load an existing Q-function on the target environment:

```bash
python src/agent/train_dqn.py --env babyai-MapSolve_0_0-v0 --output models/dqn_pretrained.pkl
```

### Query Timing Algorithm

Run the query scheduling module to compute optimal query times based on EVOI prediction and reaction time models:

```bash
python src/query_timing/run_query_timing.py \
  --model models/dqn_pretrained.pkl \
  --env highway \
  --horizon 10 \
  --threshold 0.1 \
  --reaction-time-model data/reaction_times/gamma_model.pkl
```

### Human Reaction Time Fitting

Fit statistical models to human reaction time data:

```bash
python src/human_model/fit_reaction_time.py \
  --input data/reaction_times/raw_responses.csv \
  --output data/reaction_times/fitted_models.pkl
```

## Experiments

### Human User Study

The pilot user study involved 7 participants in a highway driving simulator, measuring reaction times across repeated trials. Mean (µ = 2.178 s) and standard deviation (σ = 0.8245 s) were computed, and both Gaussian and Gamma distributions were fitted to the data citeturn0file0.

### Reaction Time Models Comparison

Driver reaction time models from literature (e.g., Kusano & Gabler, 2012) were adopted for autonomous braking scenarios. Comparison highlights that literature-derived models tend to be faster than our pilot study, likely due to participant familiarity and measurement precision citeturn0file0.

## Results and Artifacts

* EVOI curves and optimal query points for various environments are stored in `results/`.
* Demonstration videos of user experiments and simulation runs are in `videos/`.
* Interactive API documentation available under `docs/api.md`.

## Future Work

* Implement computationally efficient irrationality models for richer human behavior simulation.
* Develop a tractable and scalable EVOI prediction algorithm for high-dimensional tasks.
* Extend evaluation to robotic simulators such as MetaWorld, DM Control, and CARLA.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Masoud Jafaripour ([jafaripo@ualberta.ca](mailto:jafaripo@ualberta.ca))

School of Computing Science, University of Alberta
Edmonton, Canada
