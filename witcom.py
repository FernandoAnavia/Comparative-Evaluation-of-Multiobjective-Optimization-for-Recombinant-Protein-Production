import time
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.stats import friedmanchisquare, wilcoxon

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.asf import ASF
from pymoo.decomposition.aasf import AASF


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

N_GEN = 100
POP_SIZE = 100

PRINT_EVERY = 100
FRONT_GENERATIONS = [100]

PENALTY = 1e6

USE_OBJECTIVE_SCALING = True
PRODUCTIVITY_SCALE = 1.20
YIELD_SCALE = 0.36

SAVE_FILES = True          # <-- Activar guardado
MAKE_PLOTS = True

N_RUNS = 30

PRIME_SEEDS = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113
]

ALL_ALGORITHMS = [
    "NSGA2",
    "MOEAD_TCH",
    "MOEAD_PBI",
    "MOEAD_WS",
    "MOEAD_ASF",
    "MOEAD_AASF"
]

ALGORITHMS_TO_RUN = ALL_ALGORITHMS

ALGORITHM_COLORS = {
    "NSGA2": "blue",
    "MOEAD_WS": "darkorange",
    "MOEAD_TCH": "green",
    "MOEAD_PBI": "purple",
    "MOEAD_ASF": "red",
    "MOEAD_AASF": "brown"
}


# ============================================================
# PARÁMETROS ORIGINALES DEL MODELO
# ============================================================

kg = 1.53
Kg = 0.09
theta_a = 0.52

ka = 0.97
Ka = 0.5

g = 0.25
kover = 0.17
l = 0.7

Yg = 0.44
Ya = 0.30

beta = 0.26

kAcs = 1.46
KAcs = 0.012

kdeg = 0.0044
Gin = 20.0


# ============================================================
# FUNCIONES BIOLÓGICAS ORIGINALES
# ============================================================

def rgp(G, A):
    return kg * G / (G + Kg) * theta_a / (A + theta_a)


def rgc(G, A):
    return beta * rgp(G, A)


def downreg(y):
    phi = g / l
    return max(0.0, (1.0 + phi) * g / (g + y) - phi)


def rover_p(G, A):
    return kover * max(0.0, rgp(G, A) - l)


def raup_p(G, A):
    if A <= 0:
        return 0.0
    return ka * A / (A + Ka) * downreg(rgp(G, A))


def raup_c(G, A):
    if A <= 0:
        return 0.0

    return (
        ka * A / (A + Ka) * downreg(rgc(G, A))
        + kAcs * A / (A + KAcs)
    )


def fp(G, A):
    return Yg * rgp(G, A) - Ya * rover_p(G, A) + Ya * raup_p(G, A)


def fc(G, A):
    return Yg * rgc(G, A) + Ya * raup_c(G, A)


# ============================================================
# EQUILIBRIOS AUXILIARES ORIGINALES
# ============================================================

def producer_equilibrium(Yhp, D):
    def equations(z):
        Bp, G, A = z

        return [
            (1 - Yhp) * fp(G, A) - kdeg - D,
            D * (Gin - G) - rgp(G, A) * Bp,
            -D * A + (rover_p(G, A) - raup_p(G, A)) * Bp
        ]

    sol = least_squares(
        equations,
        x0=[1.0, Gin / 2, 0.5],
        bounds=([1e-9, 1e-9, 0.0], [1000, Gin, 1000]),
        max_nfev=250
    )

    if not sol.success or np.linalg.norm(sol.fun) > 1e-5:
        return None

    return sol.x


def cleaner_glucose_equilibrium(Yhc, D):
    Dc = (1 - Yhc) * fc(Gin, 0.0) - kdeg

    if D >= Dc:
        return Gin

    def equation(G):
        return (1 - Yhc) * fc(G[0], 0.0) - kdeg - D

    sol = least_squares(
        equation,
        x0=[Gin / 2],
        bounds=([1e-9], [Gin]),
        max_nfev=150
    )

    if not sol.success or abs(sol.fun[0]) > 1e-5:
        return None

    return sol.x[0]


# ============================================================
# RESTRICCIONES ORIGINALES
# ============================================================

def constraint_values(Yhp, Yhc, D):
    Da = (1 - Yhp) * Yg * l - kdeg
    Dp = (1 - Yhp) * fp(Gin, 0.0) - kdeg

    c1 = D - Da
    c2 = Dp - D

    prod_eq = producer_equilibrium(Yhp, D)

    if prod_eq is None:
        return np.array([c1, c2, -1, -1])

    _, Gp, Ap = prod_eq

    Gc = cleaner_glucose_equilibrium(Yhc, D)

    if Gc is None:
        return np.array([c1, c2, -1, -1])

    c3 = (1 - Yhp) * fp(Gc, 0.0) - kdeg - D
    c4 = (1 - Yhc) * fc(Gp, Ap) - kdeg - D

    return np.array([c1, c2, c3, c4])


def is_feasible(Yhp, Yhc, D):
    if not (0 <= Yhp < 0.999 and 0 <= Yhc < 0.999 and D > 0):
        return False

    return np.all(constraint_values(Yhp, Yhc, D) > 0)


# ============================================================
# EQUILIBRIO DE COEXISTENCIA ORIGINAL
# ============================================================

def coexistence_equilibrium(Yhp, Yhc, D):
    if not is_feasible(Yhp, Yhc, D):
        return None

    def equations(z):
        Bp, Bc, G, A = z

        return [
            (1 - Yhp) * fp(G, A) - kdeg - D,
            (1 - Yhc) * fc(G, A) - kdeg - D,
            D * (Gin - G) - rgp(G, A) * Bp - rgc(G, A) * Bc,
            -D * A + (rover_p(G, A) - raup_p(G, A)) * Bp - raup_c(G, A) * Bc
        ]

    sol = least_squares(
        equations,
        x0=[1.0, 1.0, Gin / 2, 0.5],
        bounds=([1e-9, 1e-9, 1e-9, 0.0], [1000, 1000, Gin, 1000]),
        max_nfev=400
    )

    if not sol.success or np.linalg.norm(sol.fun) > 1e-5:
        return None

    return sol.x


# ============================================================
# OBJETIVOS ORIGINALES
# ============================================================

def evaluate_solution(Yhp, Yhc, D):
    eq = coexistence_equilibrium(Yhp, Yhc, D)

    if eq is None:
        return None

    Bp, Bc, G, A = eq

    Hp = (Yhp / (1 - Yhp)) * Bp
    Hc = (Yhc / (1 - Yhc)) * Bc
    H_total = Hp + Hc

    productivity = D * H_total
    process_yield = H_total / Gin

    return productivity, process_yield, Bp, Bc, G, A, Hp, Hc, H_total


# ============================================================
# PROBLEMA PYMOO ORIGINAL
# ============================================================

class ProteinConsortiumProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=3,
            n_obj=2,
            xl=np.array([0.0, 0.0, 0.001]),
            xu=np.array([0.98, 0.98, 1.20])
        )

        self.n_calls = 0
        self.n_feasible = 0
        self.n_infeasible = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.n_calls += 1

        result = evaluate_solution(*x)

        if result is None:
            self.n_infeasible += 1
            out["F"] = [PENALTY, PENALTY]
            return

        self.n_feasible += 1

        productivity, process_yield, *_ = result

        if USE_OBJECTIVE_SCALING:
            out["F"] = [
                -productivity / PRODUCTIVITY_SCALE,
                -process_yield / YIELD_SCALE
            ]
        else:
            out["F"] = [-productivity, -process_yield]


class FeasibleSampling(Sampling):

    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng(self.seed)
        X = []

        attempts = 0
        max_attempts = n_samples * 5000

        while len(X) < n_samples and attempts < max_attempts:
            attempts += 1

            Yhp = rng.uniform(0.0, 0.98)
            Yhc = rng.uniform(0.0, 0.98)

            Da = (1 - Yhp) * Yg * l - kdeg
            Dp = (1 - Yhp) * fp(Gin, 0.0) - kdeg

            D_min = max(0.001, Da + 1e-5)
            D_max = min(1.20, Dp - 1e-5)

            if D_min >= D_max:
                continue

            D = rng.uniform(D_min, D_max)

            if is_feasible(Yhp, Yhc, D):
                X.append([Yhp, Yhc, D])

        while len(X) < n_samples:
            X.append([
                rng.uniform(0.0, 0.98),
                rng.uniform(0.0, 0.98),
                rng.uniform(0.001, 1.20)
            ])

        return np.array(X)


# ============================================================
# INDICADORES: HV, Spacing, IGD+, Epsilon
# ============================================================

def to_positive(F):
    if len(F) == 0:
        return np.empty((0, 2))

    if USE_OBJECTIVE_SCALING:
        return np.column_stack([
            -F[:, 0] * PRODUCTIVITY_SCALE,
            -F[:, 1] * YIELD_SCALE
        ])

    return -F


def unique_objective_points(points, decimals=8):
    if len(points) == 0:
        return points

    rounded = np.round(points, decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    idx = np.sort(idx)

    return points[idx]


def non_dominated_min(F):
    valid = np.all(F < 1e5, axis=1)
    Fv = F[valid]

    if len(Fv) == 0:
        return np.empty((0, 2)), valid

    Fv = unique_objective_points(Fv)

    idx = NonDominatedSorting().do(
        Fv,
        only_non_dominated_front=True
    )

    return Fv[idx], valid


def non_dominated_max(points):
    if len(points) == 0:
        return np.empty((0, 2))

    points = unique_objective_points(points)

    idx = NonDominatedSorting().do(
        -points,
        only_non_dominated_front=True
    )

    return points[idx]


def hypervolume_2d(points, ref=(0.0, 0.0)):
    if len(points) == 0:
        return 0.0

    pts = np.array(points, dtype=float)

    pts = pts[
        (pts[:, 0] > ref[0]) &
        (pts[:, 1] > ref[1])
    ]

    if len(pts) == 0:
        return 0.0

    pts = non_dominated_max(pts)
    pts = pts[np.argsort(pts[:, 0])]

    hv = 0.0
    previous_x = ref[0]

    for x, y in pts:
        width = x - previous_x
        height = y - ref[1]

        if width > 0 and height > 0:
            hv += width * height
            previous_x = x

    return float(hv)


def spacing_metric(points):
    if len(points) < 3:
        return np.nan

    pts = points[np.argsort(points[:, 0])]
    distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)

    return float(np.std(distances))


def igd_plus(A, Z):
    """
    IGD+ entre el conjunto A (aproximación) y el conjunto Z (referencia).
    Ambos deben estar en espacio de minimización (objetivos negados).
    """
    if len(Z) == 0 or len(A) == 0:
        return np.inf

    diff = Z[None, :, :] - A[:, None, :]
    diff = np.maximum(0.0, diff)
    dist = np.sqrt(np.sum(diff**2, axis=2))
    min_dist = np.min(dist, axis=0)
    return np.mean(min_dist)


def epsilon_indicator(A, Z):
    """
    Epsilon indicador aditivo (minimización).
    A: aproximación, Z: referencia.
    Devuelve el valor mínimo de epsilon tal que A epsilon-domina a Z.
    """
    if len(Z) == 0 or len(A) == 0:
        return np.inf

    delta = A[:, None, :] - Z[None, :, :]
    max_delta = np.max(delta, axis=2)
    min_max_delta = np.min(max_delta, axis=0)
    return np.max(min_max_delta)


def compute_reference_front(all_fronts):
    """Construye el frente de referencia no dominado a partir de una lista de frentes (maximización)."""
    all_points = []
    for front in all_fronts:
        if len(front) > 0:
            all_points.append(front)
    if not all_points:
        return np.empty((0, 2))
    pts = np.vstack(all_points)
    # Convertir a minimización para usar NonDominatedSorting
    pts_neg = -pts
    idx = NonDominatedSorting().do(pts_neg, only_non_dominated_front=True)
    return pts[idx]


# ============================================================
# CALLBACK
# ============================================================

class SimpleCallback(Callback):

    def __init__(self, algorithm_name):
        super().__init__()
        self.algorithm_name = algorithm_name
        self.history = {}
        self.metrics = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")

        if gen in FRONT_GENERATIONS:
            self.history[gen] = {
                "X": X.copy(),
                "F": F.copy()
            }

        if gen % PRINT_EVERY == 0 or gen == 1 or gen == N_GEN:
            F_nd, _ = non_dominated_min(F)
            obj_nd = to_positive(F_nd)

            hv = hypervolume_2d(obj_nd, ref=(0.0, 0.0))

            try:
                n_eval = algorithm.evaluator.n_eval
            except Exception:
                n_eval = np.nan

            self.metrics.append({
                "algorithm": self.algorithm_name,
                "generation": gen,
                "HV": hv,
                "Spacing": spacing_metric(obj_nd),
                "n_non_dominated": len(obj_nd),
                "n_eval": n_eval,
                "front": obj_nd.copy()
            })


# ============================================================
# ALGORITMOS
# ============================================================

def make_algorithm(algorithm_name, seed):
    sampling = FeasibleSampling(seed)

    if algorithm_name == "NSGA2":
        crossover = SBX(prob=0.807, eta=25.0751)
        mutation = PM(prob=0.5835, eta=23.2494)

        return NSGA2(
            pop_size=POP_SIZE,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )

    ref_dirs = get_reference_directions(
        "uniform",
        2,
        n_partitions=POP_SIZE - 1
    )

    if algorithm_name == "MOEAD_WS":
        decomposition = WeightedSum()
        n_neighbors = 17
        prob_neighbor_mating = 0.5835
        crossover = SBX(prob=0.6139, eta=27.1205)
        mutation = PM(prob=0.9012, eta=45.3323)

    elif algorithm_name == "MOEAD_TCH":
        decomposition = Tchebicheff()
        n_neighbors = 28
        prob_neighbor_mating = 0.5207
        crossover = SBX(prob=0.8090, eta=6.4873)
        mutation = PM(prob=0.0900, eta=23.3953)

    elif algorithm_name == "MOEAD_PBI":
        decomposition = PBI(theta=9.7921)
        n_neighbors = 28
        prob_neighbor_mating = 0.4115
        crossover = SBX(prob=0.9010, eta=33.0215)
        mutation = PM(prob=0.4776, eta=35.2930)

    elif algorithm_name == "MOEAD_ASF":
        decomposition = ASF()
        n_neighbors = 10
        prob_neighbor_mating = 0.4333
        crossover = SBX(prob=0.8513, eta=43.1403)
        mutation = PM(prob=0.6915, eta=24.1052)

    elif algorithm_name == "MOEAD_AASF":
        decomposition = AASF(beta=7.0580)
        n_neighbors = 5
        prob_neighbor_mating = 0.4266
        crossover = SBX(prob=0.7722, eta=37.1383)
        mutation = PM(prob=0.8050, eta=23.8340)

    else:
        raise ValueError(f"Algoritmo no reconocido: {algorithm_name}")

    return MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        prob_neighbor_mating=prob_neighbor_mating,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        decomposition=decomposition
    )


# ============================================================
# TABLA DE SOLUCIONES
# ============================================================

def build_solution_table(X, algorithm_name, seed):
    rows = []

    for x in X:
        r = evaluate_solution(*x)

        if r is None:
            continue

        productivity, process_yield, Bp, Bc, G, A, Hp, Hc, H_total = r

        rows.append({
            "algorithm": algorithm_name,
            "seed": seed,
            "Yhp": x[0],
            "Yhc": x[1],
            "D": x[2],
            "productivity_DH": productivity,
            "process_yield_H_Gin": process_yield,
            "Bp": Bp,
            "Bc": Bc,
            "G": G,
            "A": A,
            "Hp": Hp,
            "Hc": Hc,
            "H_total": H_total
        })

    return pd.DataFrame(rows)


# ============================================================
# EJECUTAR UN ALGORITMO UNA VEZ
# ============================================================

def run_one_algorithm(algorithm_name, seed):
    problem = ProteinConsortiumProblem()
    callback = SimpleCallback(algorithm_name)
    algorithm = make_algorithm(algorithm_name, seed)

    start_time = time.perf_counter()

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", N_GEN),
        seed=seed,
        callback=callback,
        verbose=False
    )

    execution_time = time.perf_counter() - start_time

    F_final = res.pop.get("F")
    X_final = res.pop.get("X")

    valid = np.all(F_final < 1e5, axis=1)
    F_valid = F_final[valid]
    X_valid = X_final[valid]

    if len(F_valid) == 0:
        F_nd = np.empty((0, 2))
        X_nd = np.empty((0, 3))
        front = np.empty((0, 2))
    else:
        nd_idx = NonDominatedSorting().do(
            F_valid,
            only_non_dominated_front=True
        )

        F_nd = F_valid[nd_idx]
        X_nd = X_valid[nd_idx]
        front = to_positive(F_nd)

    hv = hypervolume_2d(front, ref=(0.0, 0.0))
    spacing = spacing_metric(front)

    return {
        "algorithm": algorithm_name,
        "seed": seed,
        "result": res,
        "problem": problem,
        "callback": callback,
        "time_seconds": execution_time,
        "n_function_evaluations": problem.n_calls,
        "n_feasible_evaluations": problem.n_feasible,
        "n_infeasible_evaluations": problem.n_infeasible,
        "X_nd": X_nd,
        "F_nd": F_nd,
        "front": front,
        "HV": hv,
        "Spacing": spacing,
        "n_non_dominated": len(front)
    }


# ============================================================
# 30 EJECUCIONES INDEPENDIENTES
# ============================================================

def run_all_independent_experiments():
    metric_rows = []
    solution_tables = []
    all_fronts = []   # guardaremos los frentes (maximización) para referencia interna

    for alg in ALGORITHMS_TO_RUN:
        for seed in PRIME_SEEDS[:N_RUNS]:

            result = run_one_algorithm(alg, seed)

            hv = result["HV"]

            if hv > 0.8:
                print(
                    "\nADVERTENCIA: HV fuera de escala.\n"
                    f"Algoritmo={alg}, seed={seed}, HV={hv:.9f}\n"
                    "El HV esperado del código original está cerca de 0.35.\n"
                    "Revisa que no se haya cambiado el cálculo de objetivos o HV.\n"
                )

            # Guardamos el frente de esta corrida (maximización)
            front = result["front"]
            all_fronts.append(front)

            metric_rows.append({
                "algorithm": alg,
                "seed": seed,
                "HV": result["HV"],
                "Spacing": result["Spacing"],
                "n_non_dominated": result["n_non_dominated"],
                "time_seconds": result["time_seconds"],
                "n_function_evaluations": result["n_function_evaluations"],
                "n_feasible_evaluations": result["n_feasible_evaluations"],
                "n_infeasible_evaluations": result["n_infeasible_evaluations"]
                # IGD+ y Epsilon se añadirán después
            })

            df_solutions = build_solution_table(result["X_nd"], alg, seed)
            solution_tables.append(df_solutions)

            print(
                f"{alg:12s} | seed={seed:3d} | "
                f"HV={result['HV']:.9f} | "
                f"Spacing={result['Spacing']:.9f} | "
                f"ND={result['n_non_dominated']:3d} | "
                f"Time={result['time_seconds']:.2f}s | "
                f"Eval={result['n_function_evaluations']}"
            )

    # Calcular referencia interna de Python (usando todos los frentes obtenidos)
    ref_front = compute_reference_front(all_fronts)
    ref_neg = -ref_front

    # Añadir IGD+ y Epsilon a cada fila de métricas
    for i, row in enumerate(metric_rows):
        A = all_fronts[i]  # frente de esa corrida (maximización)
        if len(A) > 0 and len(ref_front) > 0:
            A_neg = -A
            row["IGD_plus"] = igd_plus(A_neg, ref_neg)
            row["Epsilon"] = epsilon_indicator(A_neg, ref_neg)
        else:
            row["IGD_plus"] = np.inf
            row["Epsilon"] = np.inf

    metrics_df = pd.DataFrame(metric_rows)

    if len(solution_tables) > 0:
        solutions_df = pd.concat(solution_tables, ignore_index=True)
    else:
        solutions_df = pd.DataFrame()

    return metrics_df, solutions_df


# ============================================================
# ESTADÍSTICA
# ============================================================

def summarize_metrics(metrics_df):
    return metrics_df.groupby("algorithm").agg(
        runs=("seed", "count"),

        HV_mean=("HV", "mean"),
        HV_std=("HV", "std"),
        HV_median=("HV", "median"),
        HV_best=("HV", "max"),
        HV_worst=("HV", "min"),

        Spacing_mean=("Spacing", "mean"),
        Spacing_std=("Spacing", "std"),
        Spacing_median=("Spacing", "median"),
        Spacing_best=("Spacing", "min"),
        Spacing_worst=("Spacing", "max"),

        ND_mean=("n_non_dominated", "mean"),
        ND_std=("n_non_dominated", "std"),

        time_mean=("time_seconds", "mean"),
        time_std=("time_seconds", "std"),

        eval_mean=("n_function_evaluations", "mean"),
        eval_std=("n_function_evaluations", "std"),

        feasible_eval_mean=("n_feasible_evaluations", "mean"),
        infeasible_eval_mean=("n_infeasible_evaluations", "mean"),

        IGD_plus_mean=("IGD_plus", "mean"),
        IGD_plus_std=("IGD_plus", "std"),
        Epsilon_mean=("Epsilon", "mean"),
        Epsilon_std=("Epsilon", "std")
    ).reset_index()


def metric_matrix(metrics_df, metric):
    algorithms = sorted(metrics_df["algorithm"].unique())
    seeds = sorted(metrics_df["seed"].unique())

    matrix = []

    for seed in seeds:
        row = []
        complete = True

        for alg in algorithms:
            values = metrics_df[
                (metrics_df["algorithm"] == alg) &
                (metrics_df["seed"] == seed)
            ][metric].values

            if len(values) == 0:
                complete = False
                break

            row.append(float(values[0]))

        if complete:
            matrix.append(row)

    return algorithms, np.array(matrix, dtype=float)


def friedman_analysis(metrics_df, metric):
    algorithms, matrix = metric_matrix(metrics_df, metric)

    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        return {
            "metric": metric,
            "statistic": np.nan,
            "p_value": np.nan,
            "status": "Datos insuficientes"
        }

    stat, p_value = friedmanchisquare(
        *[matrix[:, i] for i in range(matrix.shape[1])]
    )

    return {
        "metric": metric,
        "statistic": float(stat),
        "p_value": float(p_value),
        "status": "OK"
    }


def wilcoxon_vs_reference(metrics_df, metric, reference="NSGA2"):
    rows = []

    algorithms = sorted(metrics_df["algorithm"].unique())

    for alg in algorithms:
        if alg == reference:
            continue

        left = metrics_df[metrics_df["algorithm"] == reference][["seed", metric]]
        right = metrics_df[metrics_df["algorithm"] == alg][["seed", metric]]

        merged = pd.merge(
            left,
            right,
            on="seed",
            suffixes=("_reference", "_algorithm")
        )

        if len(merged) < 2:
            rows.append({
                "metric": metric,
                "reference": reference,
                "algorithm": alg,
                "statistic": np.nan,
                "p_value": np.nan,
                "status": "Datos insuficientes"
            })
            continue

        try:
            stat, p_value = wilcoxon(
                merged[f"{metric}_reference"],
                merged[f"{metric}_algorithm"],
                zero_method="wilcox",
                alternative="two-sided"
            )

            rows.append({
                "metric": metric,
                "reference": reference,
                "algorithm": alg,
                "statistic": float(stat),
                "p_value": float(p_value),
                "status": "OK"
            })

        except Exception as e:
            rows.append({
                "metric": metric,
                "reference": reference,
                "algorithm": alg,
                "statistic": np.nan,
                "p_value": np.nan,
                "status": str(e)
            })

    return pd.DataFrame(rows)


def holm_correction_per_metric(wilcoxon_df):
    all_rows = []

    for metric in wilcoxon_df["metric"].unique():
        sub = wilcoxon_df[
            (wilcoxon_df["metric"] == metric) &
            (wilcoxon_df["p_value"].notna())
        ].copy()

        sub = sub.sort_values("p_value").reset_index(drop=True)

        m = len(sub)

        adjusted = []
        threshold = []
        reject = []

        for i, row in sub.iterrows():
            denominator = m - i
            holm_threshold = 0.05 / denominator
            adjusted_p = min(row["p_value"] * denominator, 1.0)

            threshold.append(holm_threshold)
            adjusted.append(adjusted_p)
            reject.append(row["p_value"] <= holm_threshold)

        sub["holm_threshold"] = threshold
        sub["holm_adjusted_p"] = adjusted
        sub["holm_reject_0.05"] = reject

        all_rows.append(sub)

    if len(all_rows) == 0:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def run_statistics(metrics_df):
    test_metrics = ["HV", "Spacing", "n_non_dominated", "time_seconds",
                    "IGD_plus", "Epsilon"]

    friedman_rows = []

    for metric in test_metrics:
        friedman_rows.append(friedman_analysis(metrics_df, metric))

    friedman_df = pd.DataFrame(friedman_rows)

    wilcoxon_frames = []

    for metric in test_metrics:
        wilcoxon_frames.append(
            wilcoxon_vs_reference(metrics_df, metric, reference="NSGA2")
        )

    wilcoxon_df = pd.concat(wilcoxon_frames, ignore_index=True)

    holm_df = holm_correction_per_metric(wilcoxon_df)

    return friedman_df, wilcoxon_df, holm_df


# ============================================================
# GRÁFICA OPCIONAL
# ============================================================

def plot_boxplots(metrics_df):
    metrics = [
        ("HV", "Hypervolume"),
        ("Spacing", "Spacing"),
        ("n_non_dominated", "Non-dominated solutions"),
        ("time_seconds", "Runtime seconds"),
        ("IGD_plus", "IGD+"),
        ("Epsilon", "Epsilon indicator")
    ]

    for metric, label in metrics:
        plt.figure(figsize=(10, 5))
        data = [
            metrics_df[metrics_df["algorithm"] == alg][metric].dropna().values
            for alg in sorted(metrics_df["algorithm"].unique())
        ]
        labels = sorted(metrics_df["algorithm"].unique())

        plt.boxplot(data, labels=labels, showmeans=True)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(label)
        plt.title(label)
        plt.tight_layout()
        plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore")

    print("\nEjecutando 30 corridas independientes con IGD+ y epsilon.")
    print("Se conserva el modelo original, HV original y spacing original.\n")

    metrics_df, solutions_df = run_all_independent_experiments()

    summary_df = summarize_metrics(metrics_df)
    friedman_df, wilcoxon_df, holm_df = run_statistics(metrics_df)

    print("\n================ MÉTRICAS POR CORRIDA ================\n")
    print(metrics_df)

    print("\n================ RESUMEN MEDIA ± DESVIACIÓN ================\n")
    print(summary_df)

    print("\n================ FRIEDMAN ================\n")
    print(friedman_df)

    print("\n================ WILCOXON VS NSGA-II ================\n")
    print(wilcoxon_df)

    print("\n================ HOLM POST-HOC ================\n")
    print(holm_df)

    if MAKE_PLOTS:
        plot_boxplots(metrics_df)

    if SAVE_FILES:
        metrics_df.to_csv("python_metrics_30runs.csv", index=False)
        summary_df.to_csv("python_summary_30runs.csv", index=False)
        friedman_df.to_csv("python_friedman.csv", index=False)
        wilcoxon_df.to_csv("python_wilcoxon_vs_nsga2.csv", index=False)
        holm_df.to_csv("python_holm.csv", index=False)
        solutions_df.to_csv("python_solutions_30runs.csv", index=False)

    return metrics_df, summary_df, friedman_df, wilcoxon_df, holm_df, solutions_df


if __name__ == "__main__":
    main()