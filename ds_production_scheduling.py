
"""
Data Science Approach to AI Production Scheduling - OptiFab
============================================================
Streamlined implementation using pandas, numpy, and ML libraries
Focus: Efficient data manipulation and vectorized operations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

print("="*60)
print("DATA SCIENCE APPROACH TO AI PRODUCTION SCHEDULING")
print("="*60)

# ============================================================================
# DATA SETUP AND CONFIGURATION
# ============================================================================

class OptiFabData:
    """Centralized data container using pandas DataFrames"""

    def __init__(self):
        # Machine specifications
        self.machines = pd.DataFrame({
            'machine_id': ['M1', 'M2', 'M3'],
            'max_hours_day': [16, 16, 16],
            'shifts_day': [2, 2, 2],
            'breakdown_prob': [0.05, 0.07, 0.04],
            'downtime_hrs': [2.0, 3.0, 1.5]
        })

        # Product routing and processing times
        self.products = pd.DataFrame({
            'product_id': ['P1', 'P1', 'P2', 'P2', 'P3', 'P3', 'P4', 'P4', 'P4'],
            'machine_id': ['M1', 'M2', 'M1', 'M3', 'M2', 'M3', 'M1', 'M2', 'M3'],
            'sequence_pos': [0, 1, 0, 1, 0, 1, 0, 1, 2],
            'proc_time_unit': [2, 3, 3, 2, 4, 1, 1, 2, 2],
            'setup_time': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        })

        # Customer orders
        self.orders = pd.DataFrame({
            'order_id': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6'],
            'product_id': ['P1', 'P2', 'P3', 'P4', 'P1', 'P2'],
            'quantity': [30, 20, 25, 15, 10, 12],
            'due_date': [3, 4, 5, 6, 2, 7],
            'penalty_day': [2000, 1500, 2500, 3000, 5000, 1000]
        })

        # Generate jobs DataFrame
        self.jobs = self._create_jobs_dataframe()

        # Add derived features
        self._add_features()

        print(f"Data loaded: {len(self.jobs)} jobs, {len(self.orders)} orders")

    def _create_jobs_dataframe(self) -> pd.DataFrame:
        """Create jobs DataFrame by merging orders and products"""
        jobs_list = []
        job_id = 1

        for _, order in self.orders.iterrows():
            product_ops = self.products[self.products['product_id'] == order['product_id']]

            for _, op in product_ops.iterrows():
                jobs_list.append({
                    'job_id': f'J{job_id}',
                    'order_id': order['order_id'],
                    'product_id': order['product_id'],
                    'machine_id': op['machine_id'],
                    'sequence_pos': op['sequence_pos'],
                    'quantity': order['quantity'],
                    'proc_time_unit': op['proc_time_unit'],
                    'setup_time': op['setup_time'],
                    'due_date': order['due_date'],
                    'penalty_day': order['penalty_day']
                })
                job_id += 1

        return pd.DataFrame(jobs_list)

    def _add_features(self):
        """Add derived features using vectorized operations"""
        # Total processing time
        self.jobs['total_proc_time'] = self.jobs['quantity'] * self.jobs['proc_time_unit'] + self.jobs['setup_time']

        # Priority scores
        self.jobs['urgency_score'] = self.jobs['penalty_day'] / self.jobs['due_date']
        self.jobs['efficiency_score'] = 1 / self.jobs['total_proc_time']

        # Normalized features for ML
        scaler = StandardScaler()
        numeric_cols = ['total_proc_time', 'urgency_score', 'due_date', 'penalty_day']
        self.jobs[['proc_time_norm', 'urgency_norm', 'due_date_norm', 'penalty_norm']] = scaler.fit_transform(
            self.jobs[numeric_cols]
        )

        print("Features engineered using vectorized operations")

# ============================================================================
# VECTORIZED SCHEDULE EVALUATION
# ============================================================================

class ScheduleEvaluator:
    """Fast schedule evaluation using pandas operations"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {'makespan': 0.3, 'tardiness': 0.4, 'utilization': 0.2, 'setup': 0.1}

    def evaluate(self, schedule_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate schedule using vectorized pandas operations"""
        # Makespan
        makespan = schedule_df['end_time'].max()

        # Tardiness (vectorized calculation)
        order_completion = schedule_df.groupby('order_id')['end_time'].max()
        due_times = schedule_df.groupby('order_id')['due_date'].first() * 24
        tardiness_days = np.maximum(0, (order_completion - due_times) / 24)
        penalties = schedule_df.groupby('order_id')['penalty_day'].first()
        total_tardiness = (tardiness_days * penalties).sum()

        # Machine utilization (vectorized)
        machine_work_time = schedule_df.groupby('machine_id')['duration'].sum()
        avg_utilization = machine_work_time.mean() / makespan if makespan > 0 else 0

        # Setup time
        setup_time = schedule_df['setup_time'].sum()

        # Weighted fitness
        fitness = (
            self.weights['makespan'] * makespan +
            self.weights['tardiness'] * total_tardiness * 0.001 +
            self.weights['utilization'] * (1 - avg_utilization) * 1000 +
            self.weights['setup'] * setup_time
        )

        return {
            'makespan': makespan,
            'tardiness': total_tardiness,
            'utilization': avg_utilization,
            'setup_time': setup_time,
            'fitness': fitness
        }

# ============================================================================
# DATA-DRIVEN SCHEDULING ALGORITHMS
# ============================================================================

class DataScienceScheduler:
    """ML-enhanced scheduling using clustering and optimization"""

    def __init__(self, data: OptiFabData):
        self.data = data
        self.evaluator = ScheduleEvaluator()
        self.job_clusters = self._cluster_jobs()

    def _cluster_jobs(self) -> np.ndarray:
        """Cluster jobs using ML for intelligent grouping"""
        features = self.data.jobs[['proc_time_norm', 'urgency_norm', 'due_date_norm', 'penalty_norm']]
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(features)
        self.data.jobs['cluster'] = clusters

        print(f"Jobs clustered into {len(np.unique(clusters))} groups using K-means")
        return clusters

    def priority_schedule(self) -> pd.DataFrame:
        """Create schedule using data-driven priority rules"""
        jobs_df = self.data.jobs.copy()

        # Multi-criteria priority scoring
        jobs_df['priority_score'] = (
            0.4 * (1 - jobs_df['due_date_norm']) +
            0.3 * jobs_df['urgency_norm'] +
            0.2 * (1 - jobs_df['proc_time_norm']) +
            0.1 * jobs_df['penalty_norm']
        )

        # Sort by priority and cluster
        jobs_df = jobs_df.sort_values(['cluster', 'priority_score'], ascending=[True, False])

        return self._build_schedule(jobs_df)

    def genetic_algorithm(self, pop_size: int = 30, generations: int = 50) -> pd.DataFrame:
        """Streamlined GA using numpy arrays"""
        jobs_array = self.data.jobs.copy()
        n_jobs = len(jobs_array)

        # Initialize population as job index permutations
        population = np.array([np.random.permutation(n_jobs) for _ in range(pop_size)])

        print(f"Running GA: {pop_size} individuals, {generations} generations")

        best_fitness = float('inf')
        best_schedule = None

        for gen in range(generations):
            # Evaluate population
            fitness_scores = np.array([self._evaluate_sequence(seq, jobs_array) for seq in population])

            # Track best
            gen_best_idx = np.argmin(fitness_scores)
            if fitness_scores[gen_best_idx] < best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_schedule = self._sequence_to_schedule(population[gen_best_idx], jobs_array)

            if gen % 10 == 0:
                print(f"Gen {gen:2d}: Best={best_fitness:.1f}, Avg={fitness_scores.mean():.1f}")

            # Selection and reproduction (vectorized)
            new_population = self._evolve_population(population, fitness_scores)
            population = new_population

        return best_schedule

    def _evaluate_sequence(self, sequence: np.ndarray, jobs_df: pd.DataFrame) -> float:
        """Fast sequence evaluation"""
        schedule = self._sequence_to_schedule(sequence, jobs_df)
        return self.evaluator.evaluate(schedule)['fitness']

    def _sequence_to_schedule(self, sequence: np.ndarray, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Convert job sequence to schedule DataFrame"""
        ordered_jobs = jobs_df.iloc[sequence].copy()
        return self._build_schedule(ordered_jobs)

    def _build_schedule(self, ordered_jobs: pd.DataFrame) -> pd.DataFrame:
        """Build schedule with constraint handling using pandas"""
        schedule_data = []
        machine_available = {'M1': 0.0, 'M2': 0.0, 'M3': 0.0}
        order_completion = {}

        for _, job in ordered_jobs.iterrows():
            # Precedence constraint
            precedence_time = 0
            if job['sequence_pos'] > 0:
                order_key = (job['order_id'], job['sequence_pos'] - 1)
                precedence_time = order_completion.get(order_key, 0)

            # Schedule job
            start_time = max(machine_available[job['machine_id']], precedence_time)
            end_time = start_time + job['total_proc_time']

            schedule_data.append({
                'job_id': job['job_id'],
                'order_id': job['order_id'],
                'product_id': job['product_id'],
                'machine_id': job['machine_id'],
                'start_time': start_time,
                'end_time': end_time,
                'duration': job['total_proc_time'],
                'setup_time': job['setup_time'],
                'due_date': job['due_date'],
                'penalty_day': job['penalty_day']
            })

            machine_available[job['machine_id']] = end_time
            order_completion[(job['order_id'], job['sequence_pos'])] = end_time

        return pd.DataFrame(schedule_data)

    def _evolve_population(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Vectorized GA evolution"""
        pop_size, n_jobs = population.shape
        new_pop = np.empty_like(population)

        # Elitism
        elite_idx = np.argmin(fitness)
        new_pop[0] = population[elite_idx]

        # Tournament selection and crossover
        for i in range(1, pop_size):
            # Tournament selection
            p1_idx = self._tournament_select(fitness)
            p2_idx = self._tournament_select(fitness)

            # Order crossover
            child = self._order_crossover(population[p1_idx], population[p2_idx])

            # Mutation
            if np.random.random() < 0.1:
                child = self._mutate(child)

            new_pop[i] = child

        return new_pop

    def _tournament_select(self, fitness: np.ndarray, k: int = 3) -> int:
        """Tournament selection"""
        candidates = np.random.choice(len(fitness), k, replace=False)
        return candidates[np.argmin(fitness[candidates])]

    def _order_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Order crossover for permutations"""
        size = len(p1)
        start, end = sorted(np.random.choice(size, 2, replace=False))

        child = np.full(size, -1)
        child[start:end] = p1[start:end]

        p2_filtered = [x for x in p2 if x not in child]
        child[child == -1] = p2_filtered

        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Swap mutation"""
        mutated = individual.copy()
        i, j = np.random.choice(len(mutated), 2, replace=False)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

class ResultsAnalyzer:
    """Streamlined results analysis using pandas"""

    def __init__(self):
        self.results = {}

    def add_result(self, name: str, schedule_df: pd.DataFrame, evaluator: ScheduleEvaluator):
        """Add result for comparison"""
        evaluation = evaluator.evaluate(schedule_df)

        # On-time delivery analysis
        order_completion = schedule_df.groupby('order_id')['end_time'].max()
        due_times = schedule_df.groupby('order_id')['due_date'].first() * 24
        on_time_count = (order_completion <= due_times).sum()
        on_time_percentage = (on_time_count / len(due_times)) * 100

        self.results[name] = {
            'schedule': schedule_df,
            'evaluation': evaluation,
            'on_time_pct': on_time_percentage
        }

    def print_comparison(self):
        """Print results comparison table"""
        print("\n" + "="*70)
        print("SCHEDULING METHODS COMPARISON")
        print("="*70)

        comparison_data = []
        for name, result in self.results.items():
            eval_data = result['evaluation']
            comparison_data.append({
                'Method': name,
                'Makespan_hrs': eval_data['makespan'],
                'Tardiness_$': eval_data['tardiness'],
                'Utilization_%': eval_data['utilization'] * 100,
                'OnTime_%': result['on_time_pct'],
                'Fitness': eval_data['fitness']
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(1).to_string(index=False))

        # Best performers
        print("\nBest Performers:")
        print(f"Makespan: {comparison_df.loc[comparison_df['Makespan_hrs'].idxmin(), 'Method']}")
        print(f"Tardiness: {comparison_df.loc[comparison_df['Tardiness_$'].idxmin(), 'Method']}")
        print(f"Overall: {comparison_df.loc[comparison_df['Fitness'].idxmin(), 'Method']}")

        return comparison_df

    def pareto_analysis(self):
        """Simple Pareto analysis"""
        print("\nPareto Analysis:")
        pareto_data = []
        for name, result in self.results.items():
            eval_data = result['evaluation']
            pareto_data.append({
                'Method': name,
                'Makespan': eval_data['makespan'],
                'Tardiness': eval_data['tardiness']
            })

        pareto_df = pd.DataFrame(pareto_data)
        print(pareto_df.to_string(index=False))

    def export_best_schedule(self, filename: str = "optimized_schedule.csv"):
        """Export best performing schedule"""
        if not self.results:
            return None

        best_method = min(self.results.keys(), 
                         key=lambda x: self.results[x]['evaluation']['fitness'])
        best_schedule = self.results[best_method]['schedule']

        # Add scheduling details
        export_df = best_schedule.copy()
        export_df['day'] = (export_df['start_time'] // 24) + 1
        export_df['shift'] = ((export_df['start_time'] % 24) // 8) + 1
        export_df = export_df.round(2)

        export_df.to_csv(filename, index=False)
        print(f"\nBest schedule ({best_method}) exported to {filename}")

        return export_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_data_science_scheduling():
    """Main execution function"""
    print("\nInitializing OptiFab production data...")

    # Load and prepare data
    data = OptiFabData()
    scheduler = DataScienceScheduler(data)
    analyzer = ResultsAnalyzer()
    evaluator = ScheduleEvaluator()

    print("\n1. Running Priority-Based Scheduling...")
    priority_schedule = scheduler.priority_schedule()
    analyzer.add_result("ML-Priority", priority_schedule, evaluator)

    print("\n2. Running Genetic Algorithm...")
    ga_schedule = scheduler.genetic_algorithm(pop_size=25, generations=40)
    analyzer.add_result("Genetic-Algorithm", ga_schedule, evaluator)

    print("\n3. Analyzing Results...")
    comparison_df = analyzer.print_comparison()
    analyzer.pareto_analysis()

    print("\n4. Exporting Results...")
    best_schedule = analyzer.export_best_schedule()
    
    print("\n5. Visualizing Gantt Charts...")
    plot_gantt(priority_schedule, "Priority-Based Schedule")
    plot_gantt(ga_schedule, "Genetic Algorithm Schedule")


    print(f"\nTop 10 Jobs in Optimal Schedule:")
    display_cols = ['job_id', 'order_id', 'machine_id', 'start_time', 'end_time', 'day', 'shift']
    print(best_schedule[display_cols].head(10).to_string(index=False))

    # Performance insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    if len(analyzer.results) >= 2:
        methods = list(analyzer.results.keys())
        baseline = analyzer.results[methods[0]]['evaluation']
        best = analyzer.results[methods[1]]['evaluation']

        makespan_improvement = (baseline['makespan'] - best['makespan']) / baseline['makespan'] * 100
        tardiness_improvement = (baseline['tardiness'] - best['tardiness']) / baseline['tardiness'] * 100

        print(f"• Genetic Algorithm vs Priority Method:")
        print(f"  - Makespan improved by {makespan_improvement:.1f}%")
        print(f"  - Tardiness reduced by {tardiness_improvement:.1f}%")
        print(f"• Data science approach enables efficient optimization")
        print(f"• Vectorized operations provide fast computation")

    print(f"\n✅ Data Science Production Scheduling Completed!")
    print(f"📊 Results exported to optimized_schedule.csv")

    return data, analyzer, best_schedule

# ============================================================================
# GANTT CHART VISUALIZATION
# ============================================================================

def plot_gantt(schedule_df: pd.DataFrame, title: str = "Schedule Gantt Chart"):
    """Plot Gantt chart for a given schedule"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Assign colors per machine
    colors = {'M1': 'skyblue', 'M2': 'lightgreen', 'M3': 'salmon'}

    for _, row in schedule_df.iterrows():
        ax.barh(row['machine_id'],
                row['end_time'] - row['start_time'],
                left=row['start_time'],
                color=colors.get(row['machine_id'], 'gray'),
                edgecolor='black',
                alpha=0.8)
        ax.text(row['start_time'] + (row['end_time'] - row['start_time'])/2,
                row['machine_id'],
                row['job_id'],
                ha='center', va='center', fontsize=8, color='black')

    ax.set_xlabel("Time (hrs)")
    ax.set_ylabel("Machines")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Execute the system
if __name__ == "__main__":
    print("STARTING DATA SCIENCE PRODUCTION SCHEDULING...")

    try:
        data, results, schedule = run_data_science_scheduling()
        print("\n🎉 SUCCESS: All optimizations completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Required: pip install pandas numpy matplotlib scikit-learn")
