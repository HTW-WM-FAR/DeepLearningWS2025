Deep Reinforcement Learning for Portfolio Optimization (S&P 500)

Dieses Repository enthält den Code und die Dokumentation für eine Seminararbeit im Bereich Deep Reinforcement Learning (DRL) im quantitativen Finanzwesen.

Ziel des Projekts war die Entwicklung eines PPO-Agenten (Proximal Policy Optimization), der ein Portfolio aus S&P 500 Aktien dynamisch verwaltet und dabei den Markt unter Berücksichtigung realer Transaktionskosten schlägt.

📈 Ergebnisse & Highlights

Performance: Der trainierte Agent erzielt eine leicht höhere risikoadjustierte Rendite als der Benchmark (S&P 500 Buy & Hold) im Evaluationszeitraum (2020–2023).

Architektur: Custom LSTM-Feature-Extractor mit Dropout zur Vermeidung von Overfitting.

Robustheit: Erfolgreiches Lernen von Kosteneffizienz durch Implementierung einer Turnover Penalty.

Daten: Training auf 14 Jahren echter S&P 500 Daten (2006–2019), inklusive der Finanzkrise 2008.

📂 Projektstruktur

Um das Repository performant zu halten und GitHub-Limits zu umgehen, werden große Datensätze und Modell-Checkpoints nicht direkt hochgeladen. Die Struktur ist wie folgt aufgebaut:

PPO_Portfolio_Optimization/
│
├── data/                    # Lokaler Speicher für CSV-Daten (wird durch Notebooks generiert)
│   ├── raw/                 # Rohdaten von yfinance (wird ignoriert durch .gitignore)
│   └── processed/           # Bereinigte Features (wird ignoriert durch .gitignore)
│
├── models/                  # Speicherort für trainierte Agenten (lokal)
│
├── notebooks/               # Der Kern des Projekts: 5 sequentielle Schritte
│   ├── 00_Data_Generator_Synthetic.ipynb      # Testet die Pipeline mit synthetischen Daten
│   ├── 01_Data_Downloader_Real.ipynb          # Lädt echte S&P 500 Daten herunter
│   ├── 02_Data_Validation_Features.ipynb      # Feature Engineering (RSI, MACD, VIX)
│   ├── 03_PPO_Training.ipynb                  # Training des Agenten
│   └── 04_Final_Evaluation_Quantstats.ipynb   # Backtesting und Reporting
│
├── results/                 # Output der Evaluation
│   └── reports/             # HTML-Reports von QuantStats
│
├── requirements.txt         # Benötigte Python-Bibliotheken
└── README.md                # Diese Datei


🚀 Installation & Ausführung (Reproduzierbarkeit)

1. Repository klonen

git clone [https://github.com/DEIN_USERNAME/PPO-Portfolio-Seminar.git](https://github.com/DEIN_USERNAME/PPO-Portfolio-Seminar.git)
cd PPO-Portfolio-Seminar


2. Umgebung einrichten

Es wird empfohlen, eine virtuelle Umgebung (venv oder conda) zu nutzen.

pip install -r requirements.txt


3. Pipeline ausführen

Die Notebooks sind sequentiell aufgebaut. Da die Rohdaten nicht im Repository liegen, müssen die Notebooks in dieser Reihenfolge ausgeführt werden:

01_Data_Downloader_Real.ipynb:

Lädt historische S&P 500 Daten (2006–2024) via yfinance herunter.

Speichert data/raw/sp500_20_years_data.csv.

02_Data_Validation_Features.ipynb:

Bereinigt Daten, füllt Lücken (Forward Fill) und berechnet technische Indikatoren.

Speichert data/processed/features_cleaned.csv.

03_PPO_Training.ipynb:

Startet das Training des PPO-Agenten.

Speichert das Modell unter models/.

Nutzt Weights & Biases für das Logging (optional).

04_Final_Evaluation_Quantstats.ipynb:

Lädt das beste Modell (best_model.zip).

Führt einen Backtest auf "Out-of-Sample" Daten (2020–2023) durch.

Erstellt einen HTML-Report mit Benchmark-Vergleich.

🧠 Methodik

Algorithmus: PPO (Stable Baselines 3) mit MultiInputPolicy.

Environment: Gymnasium Custom Env mit kontinuierlichem Action-Space (Gewichtung der Aktien + Cash).

Reward Function: Log-Returns mit Scaling-Faktor (x100) und Bestrafung für hohen Umsatz (Turnover Penalty), um Overtrading zu vermeiden.

Training: 30 Millionen Timesteps auf Daten von 2006 bis 2019.

Evaluation: Out-of-Sample Test auf den Jahren 2020 bis 2023 (beinhaltet Corona-Crash und Inflationsphase).

⚠️ Hinweis zu den Daten

Aufgrund der Dateigrößenbeschränkung von GitHub (>100MB) sind die Dateien sp500_20_years_data.csv und features_cleaned.csv nicht direkt im Repository enthalten.

Sie werden jedoch automatisch und deterministisch generiert, sobald Sie die Notebooks 01 und 02 ausführen. Dies stellt die Reproduzierbarkeit der Ergebnisse sicher.

📧 Kontakt
Autor: ChristophBieritz1989@googlemail.com
Universität: HTW BERLIN
Kurs: Seminar: Deep Learning