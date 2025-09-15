# 🏭 AI-Based Production Scheduling – OptiFab

This project implements **AI-based production scheduling** for a smart factory scenario called **OptiFab**.  
It demonstrates how **heuristics and AI algorithms** like **Genetic Algorithms (GA)** can optimize production schedules to minimize makespan, reduce tardiness, maximize machine utilization, and handle setup losses.

---

## 📌 Problem Overview
Production scheduling is an **NP-hard optimization problem**, where we must assign jobs to machines under constraints such as:
- Machine capacity (1 job per machine at a time)  
- Product routing sequences  
- Setup times between product changes  
- Due dates and penalties for late orders  
- Workforce availability (only 2 operators per shift → max 2 machines in parallel, planned for extension)

---

## 🛠️ Case Scenario – OptiFab
OptiFab manufactures **4 products (P1–P4)** using **3 machines (M1–M3)**.  

### Machine Data
| Machine | Max Hours/Day | Shifts | Breakdown Probability | Downtime (hrs) |
|---------|---------------|--------|------------------------|----------------|
| M1      | 16            | 2      | 5%                     | 2.0            |
| M2      | 16            | 2      | 7%                     | 3.0            |
| M3      | 16            | 2      | 4%                     | 1.5            |

### Product Routing
- **P1:** M1 → M2  
- **P2:** M1 → M3  
- **P3:** M2 → M3  
- **P4:** M1 → M2 → M3  

Each routing step has **processing times** and a **setup time of 0.5 hrs**.

### Orders
| Order | Product | Quantity | Due (days) | Penalty (₹/day late) |
|-------|---------|----------|------------|-----------------------|
| O1    | P1      | 30       | 3          | 2000                 |
| O2    | P2      | 20       | 4          | 1500                 |
| O3    | P3      | 25       | 5          | 2500                 |
| O4    | P4      | 15       | 6          | 3000                 |
| O5    | P1      | 10       | 2          | 5000                 |
| O6    | P2      | 12       | 7          | 1000                 |

---

## 🎯 Objectives
The project focuses on **multi-objective optimization**:
1. Minimize **Makespan** (completion time of all jobs)  
2. Minimize **Tardiness** (late deliveries with penalties)  
3. Maximize **Machine Utilization**  
4. Reduce **Setup Losses**  

---

## 🤖 Approaches Implemented
1. **Priority-Based Scheduling (Heuristic + ML)**  
   - Uses clustering (K-Means) to group jobs  
   - Computes multi-criteria priority scores (due dates, urgency, efficiency)  
   - Builds a feasible baseline schedule  

2. **Genetic Algorithm (GA)**  
   - Evolves job sequences through **selection, crossover, and mutation**  
   - Optimizes makespan, tardiness, and utilization simultaneously  
   - Provides a more compact and efficient schedule  

---

## 📊 Results
- **Priority Schedule:** Provides a valid baseline but suffers from idle times and higher makespan.  
- **GA Schedule:** Reduces tardiness and makespan significantly.  
- **Pareto Analysis:** Shows trade-offs between objectives.  

### Example Gantt Charts
📌 Priority-Based Schedule:  
*(longer makespan, idle times on M3)*  

📌 Genetic Algorithm Schedule:  
*(shorter makespan, better balance across machines)*  

---

## 📂 Project Structure
.
├── ds_production_scheduling.py # Main implementation
├── optimized_schedule.csv # Best schedule exported
├── AI_Production_Scheduling_Assignment.pdf # Assignment brief
├── Figure_1.png # Example Gantt chart
└── README.md # Project documentation


---

## 🚀 How to Run
1. Clone this repository  
```bash
git clone https://github.com/your-username/ai-production-scheduling.git
cd ai-production-scheduling
```
2. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn
```
3. Run the scheduler
```bash
python ds_production_scheduling.py
```
4. Outputs generated:
- Console comparison of heuristic vs GA
- Gantt charts for visualization
- optimized_schedule.csv with best schedule

## 🔮 Future Work

- Enforce workforce constraint (max 2 machines in parallel)
- Integrate machine breakdown simulation
- Deploy scheduling as a web-based dashboard
- Extend to multi-factory scheduling with cloud/edge computing

## 📜 License

This project is for academic and research purposes only.
You may adapt and extend it for learning or personal use.
