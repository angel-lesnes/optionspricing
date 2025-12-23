# Option Pricing Simulator

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<p align="center">
  <a href="https://options-pricing-simulator.streamlit.app/" style="text-decoration: none;">
    <img src="src/app/images/logo.png" height="30" style="vertical-align: middle;">
    <span style="vertical-align: middle; font-weight: bold; margin-left: 10px;">Click to view the website</span>
  </a>
</p>

An interactive dashboard to price financial options using various mathematical models and compare to market datas. 
Designed for educational purposes to analyze and interpret outputs (greeks, volatility, convergence, etc.)

## Demo

![Application Demo](src/app/images/demo.gif)

## Features

* **Black-Scholes Model :** Real-time pricing of European options with Greeks visualization ($\Delta$, $\Gamma$, $\Theta$, $\nu$, $\rho$).
* **Binomial Tree (CRR) :** Step-by-step visualization of American option pricing using Cox-Ross-Rubinstein trees.
* **Monte Carlo Simulation :** Stochastic pricing with variance reduction techniques (Antithetic & Control Variates) and convergence analysis.
* **Interactive UI :** Built with Streamlit for dynamic parameter adjustment.
* **Interactive UX :** Built with Streamlit for understanding results issues.

## Project Structure

```text
├── src/
│   ├── app/
│   │   ├── streamlit_app.py    # Application entry point
│   │   ├── data_fetcher.py     # Market data retrieval logic
│   │   └── views/              # Page layouts (Home, BS, Binomial, MC)
│   └── pricing/
│       ├── __init__.py         # Package initialization
│       ├── black_scholes.py    # Analytical formulas
│       ├── greeks.py           # Sensitivity calculations
│       ├── binomial.py         # Tree algorithms
│       └── monte_carlo.py      # Simulation logic
├── tests/                      # Unit tests for pricing logic
├── pyproject.toml              # Build system configuration
├── requirements.txt            # Project dependencies
├── LICENSE                     # Project license
└── README.md                   # Documentation
```

## Local use 

### Clone the project

git clone <https://github.com/angel-lesnes/optionspricing>
cd optionspricing

### Activate venv

For Windows :

python -m venv venv
.\venv\Scripts\activate

For MacOS / Linux :

python3 -m venv venv
source venv/bin/activate

### Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

### Launch app

streamlit run .\src\app\streamlit_app.py

### Error : "Error data: Too Many Requests. Rate limited. Try after a while."

If this error occurs, try updating yfinance using: pip install yfinance --upgrade

## Connect & Credits

<p align="center">

<strong>Created by Lesnes Angel</strong> &nbsp;&nbsp; • &nbsp;&nbsp; <a href="https://www.linkedin.com/in/angel-lesnes-7714b6386" target="_blank"> <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin" valign="middle"> </a> </p>

> [!WARNING] 
> **Educational purpose only :** This application is developed for educational and informational purposes only. The data, calculations, and models provided do not constitute financial, investment, or trading advice. Always consult a certified financial professional before making investment decisions

📜 Copyright
© 2025 Lesnes Angel. All rights reserved.