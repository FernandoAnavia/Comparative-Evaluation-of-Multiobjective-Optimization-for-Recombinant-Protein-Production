clc; close all;

SCRIPT_FULLPATH = mfilename('fullpath');

if isempty(SCRIPT_FULLPATH)
    SCRIPT_DIR = pwd;
else
    SCRIPT_DIR = fileparts(SCRIPT_FULLPATH);
end

ORIGINAL_PWD = pwd;
ORIGINAL_PATH = path;
cleanup_obj = onCleanup(@() restore_environment(ORIGINAL_PWD, ORIGINAL_PATH));

prepare_fmincon_environment(SCRIPT_DIR);


% CONFIGURACIÓN GENERAL

SAVE_FILES = true;
MAKE_PLOTS = false;

N_RUNS = 30;

PRIME_SEEDS = [ ...
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, ...
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71, ...
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113];

PRODUCTIVITY_SCALE = 1.20;
YIELD_SCALE = 0.36;

LB = [0.00, 0.00, 0.001];
UB = [0.98, 0.98, 1.20];

HV_REF = [0.0, 0.0];

CLEANER_PRODUCES_PROTEIN = true;

configs(1).name = 'FMINCON_49';
configs(1).n_weights = 49;
configs(1).n_restarts = 1;

configs(2).name = 'FMINCON_100';
configs(2).n_weights = 100;
configs(2).n_restarts = 1;

fprintf('\n============================================================\n');
fprintf('FMINCON - comparación contra Python/pymoo\n');
fprintf('============================================================\n');
fprintf('Modelo original\n');
fprintf('============================================================\n\n');


% PARÁMETROS DEL MODELO

P.kg = 1.53;
P.Kg = 0.09;
P.theta_a = 0.52;

P.ka = 0.97;
P.Ka = 0.5;

P.g = 0.25;
P.kover = 0.17;
P.l = 0.7;

P.Yg = 0.44;
P.Ya = 0.30;

P.beta = 0.26;

P.kAcs = 1.46;
P.KAcs = 0.012;

P.kdeg = 0.0044;
P.Gin = 20.0;

P.PRODUCTIVITY_SCALE = PRODUCTIVITY_SCALE;
P.YIELD_SCALE = YIELD_SCALE;
P.CLEANER_PRODUCES_PROTEIN = CLEANER_PRODUCES_PROTEIN;


% OPCIONES FMINCON

options = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'Display', 'none', ...
    'MaxIterations', 300, ...
    'MaxFunctionEvaluations', 1500, ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-7, ...
    'FiniteDifferenceType', 'central', ...
    'FiniteDifferenceStepSize', [1e-4, 1e-4, 1e-5], ...
    'ScaleProblem', 'obj-and-constr');


% PREFLIGHT

rng(2);

preflight_points = [];
preflight_objectives = [];

for k = 1:100
    x0 = random_feasible_point(P, LB, UB);

    if ~isempty(x0)
        ev = evaluate_solution(x0, P);

        if ~isempty(ev)
            preflight_points = [preflight_points; x0];
            preflight_objectives = [preflight_objectives; ev.productivity, ev.process_yield];
        end
    end
end

fprintf('Preflight: puntos factibles encontrados = %d\n', size(preflight_points,1));

if isempty(preflight_points)
    error('No se encontraron puntos factibles en preflight. Revisa modelo, límites o parámetros.');
end

preflight_nd_idx = nondominated_indices_max(preflight_objectives);
preflight_nd = preflight_objectives(preflight_nd_idx, :);
preflight_hv = hypervolume_2d_max(preflight_nd, HV_REF);

fprintf('Preflight mejor productividad: %.9f\n', max(preflight_objectives(:,1)));
fprintf('Preflight mejor rendimiento: %.9f\n', max(preflight_objectives(:,2)));
fprintf('Preflight HV banco aleatorio: %.9f\n', preflight_hv);
fprintf('Preflight no dominadas: %d\n\n', size(preflight_nd,1));


% EJECUCIÓN PRINCIPAL

all_algorithm = {};
all_seed = [];
all_HV = [];
all_Spacing = [];
all_ND = [];
all_time = [];
all_fmincon_funcCount = [];
all_obj_calls = [];
all_con_calls = [];
all_eq_calls = [];
all_successful_weight_solutions = [];
all_fmincon_runs = [];

all_solution_tables = {};
all_fronts = struct();

row_id = 0;

for c = 1:length(configs)

    config = configs(c);

    fprintf('\n============================================================\n');
    fprintf('Configuración: %s\n', config.name);
    fprintf('N_WEIGHTS = %d | N_RESTARTS = %d\n', ...
        config.n_weights, config.n_restarts);
    fprintf('============================================================\n');

    for s = 1:N_RUNS

        seed = PRIME_SEEDS(s);
        rng(seed);

        global OBJ_EVALS CON_EVALS EQ_EVALS
        OBJ_EVALS = 0;
        CON_EVALS = 0;
        EQ_EVALS = 0;

        fprintf('\n%s | seed=%d\n', config.name, seed);

        execution_tic = tic;

        weights = linspace(0, 1, config.n_weights);

        all_results = [];
        total_fmincon_funcCount = 0;
        total_fmincon_runs = 0;
        successful_weight_solutions = 0;

        previous_best_x = [];

        for iw = 1:length(weights)

            lambda = weights(iw);

            best_x = [];
            best_fval = inf;
            best_output = [];
            best_exitflag = -999;

            for r = 1:config.n_restarts

                if r == 1 && ~isempty(previous_best_x)
                    x0 = previous_best_x;
                else
                    x0 = random_feasible_point(P, LB, UB);
                end

                if isempty(x0)
                    continue;
                end

                obj_fun = @(x) scalar_objective(x, lambda, P);
                con_fun = @(x) nonlinear_constraints(x, P);

                try
                    [x_opt, fval, exitflag, output] = call_native_fmincon( ...
                        obj_fun, ...
                        x0, ...
                        [], [], [], [], ...
                        LB, UB, ...
                        con_fun, ...
                        options);

                    total_fmincon_runs = total_fmincon_runs + 1;
                    total_fmincon_funcCount = total_fmincon_funcCount + output.funcCount;

                    if exitflag > 0 && is_feasible(x_opt, P) && fval < best_fval
                        best_x = x_opt;
                        best_fval = fval;
                        best_output = output;
                        best_exitflag = exitflag;
                    end

                catch ME
                    fprintf('lambda = %.3f | restart=%d | error fmincon: %s\n', ...
                        lambda, r, ME.message);
                end
            end

            if ~isempty(best_x)

                ev = evaluate_solution(best_x, P);

                if ~isempty(ev)
                    successful_weight_solutions = successful_weight_solutions + 1;
                    previous_best_x = best_x;

                    Dc = cleaner_threshold_Dc(best_x(2), P);
                    D_less_than_Dc = double(best_x(3) < Dc);

                    if isempty(best_output)
                        func_evals = NaN;
                    else
                        func_evals = best_output.funcCount;
                    end

                    row = [
                        lambda, ...
                        best_x(1), best_x(2), best_x(3), ...
                        ev.productivity, ev.process_yield, ...
                        ev.Bp, ev.Bc, ev.G, ev.A, ...
                        ev.Hp, ev.Hc, ev.H_total, ...
                        Dc, D_less_than_Dc, ...
                        best_fval, best_exitflag, func_evals
                    ];

                    all_results = [all_results; row];
                end
            end
        end

        execution_time = toc(execution_tic);

        var_names = { ...
            'lambda', ...
            'Yhp', 'Yhc', 'D', ...
            'productivity_DH', 'process_yield_H_Gin', ...
            'Bp', 'Bc', 'G', 'A', ...
            'Hp', 'Hc', 'H_total', ...
            'Dc', 'D_less_than_Dc', ...
            'weighted_objective', 'exitflag', 'func_evals' ...
        };

        if isempty(all_results)
            warning('%s seed=%d no produjo soluciones factibles.', config.name, seed);

            T = table();
            T_nd = table();
            points_nd = [];
            HV = 0.0;
            Spacing = NaN;
            ND = 0;

        else
            T = array2table(all_results, 'VariableNames', var_names);

            points = [T.productivity_DH, T.process_yield_H_Gin];

            nd_idx = nondominated_indices_max(points);
            T_nd = T(nd_idx, :);

            points_nd = [T_nd.productivity_DH, T_nd.process_yield_H_Gin];

            rounded_points = round(points_nd * 1e8) / 1e8;
            [~, unique_idx] = unique(rounded_points, 'rows', 'stable');
            T_nd = T_nd(unique_idx, :);
            points_nd = [T_nd.productivity_DH, T_nd.process_yield_H_Gin];

            [~, sort_idx] = sort(T_nd.productivity_DH, 'ascend');
            T_nd = T_nd(sort_idx, :);
            points_nd = [T_nd.productivity_DH, T_nd.process_yield_H_Gin];

            HV = hypervolume_2d_max(points_nd, HV_REF);
            Spacing = spacing_metric_python_style(points_nd);
            ND = height(T_nd);
        end

        % Validaciones de escala
        hv_scale_limit = PRODUCTIVITY_SCALE * YIELD_SCALE * 1.10;

        if HV > hv_scale_limit
            warning(['HV fuera de escala: %.9f. ', ...
                     'Con las escalas actuales, un HV mayor que %.9f ', ...
                     'sugiere que se está calculando HV sobre objetivos escalados ', ...
                     'o que los objetivos salieron de rango.'], HV, hv_scale_limit);
        end

        if ~isempty(points_nd)
            max_prod = max(points_nd(:,1));
            max_yield = max(points_nd(:,2));

            if max_prod > 1.50 || max_yield > 0.50
                warning(['Objetivos fuera de escala: max_prod=%.6f, max_yield=%.6f. ', ...
                         'Esto puede inflar artificialmente el HV.'], max_prod, max_yield);
            end
        end

        row_id = row_id + 1;

        all_algorithm{row_id,1} = config.name;
        all_seed(row_id,1) = seed;
        all_HV(row_id,1) = HV;
        all_Spacing(row_id,1) = Spacing;
        all_ND(row_id,1) = ND;
        all_time(row_id,1) = execution_time;
        all_fmincon_funcCount(row_id,1) = total_fmincon_funcCount;
        all_obj_calls(row_id,1) = OBJ_EVALS;
        all_con_calls(row_id,1) = CON_EVALS;
        all_eq_calls(row_id,1) = EQ_EVALS;
        all_successful_weight_solutions(row_id,1) = successful_weight_solutions;
        all_fmincon_runs(row_id,1) = total_fmincon_runs;

        if ~isempty(T_nd)
            T_nd.algorithm = repmat({config.name}, height(T_nd), 1);
            T_nd.seed = repmat(seed, height(T_nd), 1);
        end

        all_solution_tables{row_id,1} = T_nd;

        all_fronts(row_id).algorithm = config.name;
        all_fronts(row_id).seed = seed;
        all_fronts(row_id).points_nd = points_nd;

        if ~isempty(T)
            all_fronts(row_id).points_all = [T.productivity_DH, T.process_yield_H_Gin];
        else
            all_fronts(row_id).points_all = [];
        end

        fprintf('%-12s | seed=%3d | HV=%.9f | Spacing=%.9f | ND=%3d | Time=%.2fs | Eval=%d | ObjCalls=%d | ConCalls=%d | EqCalls=%d | Sol=%d | Runs=%d\n', ...
            config.name, seed, HV, Spacing, ND, execution_time, ...
            total_fmincon_funcCount, OBJ_EVALS, CON_EVALS, EQ_EVALS, ...
            successful_weight_solutions, total_fmincon_runs);
    end
end

% Referencia a partir de todos los frentes
all_points = [];
for i = 1:length(all_fronts)
    if ~isempty(all_fronts(i).points_nd)
        all_points = [all_points; all_fronts(i).points_nd];
    end
end
if ~isempty(all_points)
    ref_idx = nondominated_indices_max(all_points);
    ref_front = all_points(ref_idx, :);
else
    ref_front = [];
end

all_IGD_plus = zeros(length(all_fronts), 1);
all_Epsilon = zeros(length(all_fronts), 1);

if ~isempty(ref_front)
    ref_neg = -ref_front;   % minimización
    for i = 1:length(all_fronts)
        pts = all_fronts(i).points_nd;
        if ~isempty(pts)
            A_neg = -pts;
            all_IGD_plus(i) = igd_plus(A_neg, ref_neg);
            all_Epsilon(i) = epsilon_indicator(A_neg, ref_neg);
        else
            all_IGD_plus(i) = inf;
            all_Epsilon(i) = inf;
        end
    end
else
    all_IGD_plus(:) = inf;
    all_Epsilon(:) = inf;
end


% TABLAS FINALES

metrics = table( ...
    all_algorithm, ...
    all_seed, ...
    all_HV, ...
    all_Spacing, ...
    all_ND, ...
    all_time, ...
    all_fmincon_funcCount, ...
    all_obj_calls, ...
    all_con_calls, ...
    all_eq_calls, ...
    all_successful_weight_solutions, ...
    all_fmincon_runs, ...
    all_IGD_plus, ...
    all_Epsilon, ...
    'VariableNames', { ...
        'algorithm', ...
        'seed', ...
        'HV', ...
        'Spacing', ...
        'n_non_dominated', ...
        'time_seconds', ...
        'n_function_evaluations', ...
        'n_objective_function_calls', ...
        'n_constraint_function_calls', ...
        'n_equilibrium_solver_calls', ...
        'n_successful_weight_solutions', ...
        'n_fmincon_runs', ...
        'IGD_plus', ...
        'Epsilon' ...
    } ...
);

summary = summarize_metrics(metrics);
wilcoxon_results = wilcoxon_analysis(metrics);
holm_results = holm_correction(wilcoxon_results);
friedman_results = friedman_analysis(metrics);

if ~isempty(all_solution_tables)
    nonempty = cellfun(@(x) ~isempty(x), all_solution_tables);

    if any(nonempty)
        solutions_nd = vertcat(all_solution_tables{nonempty});
    else
        solutions_nd = table();
    end
else
    solutions_nd = table();
end

fprintf('\n================ MÉTRICAS POR CORRIDA ================\n');
disp(metrics);

fprintf('\n================ RESUMEN MEDIA +/- DESVIACIÓN ================\n');
disp(summary);

fprintf('\n================ WILCOXON ================\n');
disp(wilcoxon_results);

fprintf('\n================ HOLM ================\n');
disp(holm_results);

fprintf('\n================ FRIEDMAN ================\n');
disp(friedman_results);

if SAVE_FILES
    writetable(metrics, 'matlab_fmincon_metrics_30runs.csv');
    writetable(summary, 'matlab_fmincon_summary_30runs.csv');
    writetable(wilcoxon_results, 'matlab_fmincon_wilcoxon_30runs.csv');
    writetable(holm_results, 'matlab_fmincon_holm_30runs.csv');
    writetable(friedman_results, 'matlab_fmincon_friedman_30runs.csv');

    if ~isempty(solutions_nd)
        writetable(solutions_nd, 'matlab_fmincon_solutions_nd_30runs.csv');
    end
end

if MAKE_PLOTS
    plot_fronts(all_fronts);
end

outputs = struct();
outputs.metrics = metrics;
outputs.summary = summary;
outputs.wilcoxon = wilcoxon_results;
outputs.holm = holm_results;
outputs.friedman = friedman_results;
outputs.solutions_nd = solutions_nd;
outputs.fronts = all_fronts;

assignin('base', 'outputs', outputs);
assignin('base', 'metrics', metrics);
assignin('base', 'summary', summary);


% FUNCIONES LOCALES

function prepare_fmincon_environment(script_dir)

safe_dir = fullfile(tempdir, 'witcon_fmincon_safe_workspace');

if ~exist(safe_dir, 'dir')
    mkdir(safe_dir);
end

cd(safe_dir);

try
    if ~isempty(script_dir) && exist(script_dir, 'dir')
        % Quitar del path solo si realmente está incluido
        path_cells = strsplit(path, pathsep);
        if ismember(script_dir, path_cells)
            rmpath(script_dir);
        end
    end
catch
end

rehash toolboxcache;

remove_shadowing_fmincon_paths();

rehash toolboxcache;

hits = which('fmincon', '-all');

if ischar(hits)
    hits = cellstr(hits);
end

if isempty(hits)
    error('No se encontró fmincon. Revisa que Optimization Toolbox esté instalado.');
end

first_hit = hits{1};

if ~is_toolbox_file(first_hit)
    error(['MATLAB sigue resolviendo fmincon desde una ruta no nativa:\n%s\n', ...
           'Mueve o renombra ese archivo, o quita su carpeta del path.'], first_hit);
end

end


function remove_shadowing_fmincon_paths()

hits = which('fmincon', '-all');

if ischar(hits)
    hits = cellstr(hits);
end

for i = 1:length(hits)
    hit = hits{i};

    if isempty(hit)
        continue;
    end

    if is_toolbox_file(hit)
        continue;
    end

    hit_dir = fileparts(hit);

    if exist(hit_dir, 'dir')
        try
            rmpath(hit_dir);
            fprintf('Ruta removida temporalmente por sombrear fmincon: %s\n', hit_dir);
        catch
        end
    end
end

end


function tf = is_toolbox_file(file_path)

if isempty(file_path)
    tf = false;
    return;
end

f = lower(char(file_path));
root = lower(matlabroot);

tf = startsWith(f, root) && ...
     contains(f, [filesep 'toolbox' filesep]);

end


function restore_environment(original_pwd, original_path)

try
    path(original_path);
catch
end

try
    cd(original_pwd);
catch
end

rehash toolboxcache;

end


function [x_opt, fval, exitflag, output] = call_native_fmincon( ...
    obj_fun, x0, A, b, Aeq, beq, LB, UB, con_fun, options)

[x_opt, fval, exitflag, output] = fmincon( ...
    obj_fun, x0, A, b, Aeq, beq, LB, UB, con_fun, options);

end


function f = scalar_objective(x, lambda, P)

global OBJ_EVALS
OBJ_EVALS = OBJ_EVALS + 1;

ev = evaluate_solution(x, P);

if isempty(ev)
    margins = constraint_values(x(1), x(2), x(3), P);
    violation = sum(max(0, -margins).^2);
    f = 1e4 + 1e5 * violation;
    return;
end

prod_norm = ev.productivity / P.PRODUCTIVITY_SCALE;
yield_norm = ev.process_yield / P.YIELD_SCALE;

f = -(lambda * prod_norm + (1 - lambda) * yield_norm);

if ~isfinite(f)
    f = 1e8;
end

end


function [c, ceq] = nonlinear_constraints(x, P)

global CON_EVALS
CON_EVALS = CON_EVALS + 1;

margins = constraint_values(x(1), x(2), x(3), P);

c = -margins(:);
ceq = [];

end


function x0 = random_feasible_point(P, LB, UB)

x0 = [];
max_attempts = 10000;

for k = 1:max_attempts
    Yhp = LB(1) + rand() * (UB(1) - LB(1));
    Yhc = LB(2) + rand() * (UB(2) - LB(2));

    Da = (1 - Yhp) * P.Yg * P.l - P.kdeg;
    Dp = (1 - Yhp) * fp(P.Gin, 0.0, P) - P.kdeg;

    D_min = max(LB(3), Da + 1e-5);
    D_max = min(UB(3), Dp - 1e-5);

    if D_min >= D_max
        continue;
    end

    D = D_min + rand() * (D_max - D_min);
    candidate = [Yhp, Yhc, D];

    if is_feasible(candidate, P)
        x0 = candidate;
        return;
    end
end

end


function ok = is_feasible(x, P)

Yhp = x(1);
Yhc = x(2);
D = x(3);

if ~(Yhp >= 0 && Yhp < 0.999 && Yhc >= 0 && Yhc < 0.999 && D > 0)
    ok = false;
    return;
end

margins = constraint_values(Yhp, Yhc, D, P);
ok = all(margins >= -1e-8);

end


function margins = constraint_values(Yhp, Yhc, D, P)

Da = (1 - Yhp) * P.Yg * P.l - P.kdeg;
Dp = (1 - Yhp) * fp(P.Gin, 0.0, P) - P.kdeg;

c1 = D - Da;
c2 = Dp - D;

prod_eq = producer_equilibrium(Yhp, D, P);

if isempty(prod_eq)
    margins = [c1, c2, -1, -1];
    return;
end

Gp = prod_eq(2);
Ap = prod_eq(3);

Gc = cleaner_glucose_equilibrium(Yhc, D, P);

if isempty(Gc)
    margins = [c1, c2, -1, -1];
    return;
end

c3 = (1 - Yhp) * fp(Gc, 0.0, P) - P.kdeg - D;
c4 = (1 - Yhc) * fc(Gp, Ap, P) - P.kdeg - D;

margins = [c1, c2, c3, c4];

end


function Dc = cleaner_threshold_Dc(Yhc, P)

Dc = (1 - Yhc) * fc(P.Gin, 0.0, P) - P.kdeg;

end


function sol = producer_equilibrium(Yhp, D, P)

global EQ_EVALS
EQ_EVALS = EQ_EVALS + 1;

fun = @(z) [
    (1 - Yhp) * fp(z(2), z(3), P) - P.kdeg - D;
    D * (P.Gin - z(2)) - rgp(z(2), z(3), P) * z(1);
    -D * z(3) + (rover_p(z(2), z(3), P) - raup_p(z(2), z(3), P)) * z(1)
];

z0 = [1.0, P.Gin / 2, 0.5];
lb = [1e-9, 1e-9, 0.0];
ub = [1000, P.Gin, 1000];

opts = optimoptions('lsqnonlin', ...
    'Display', 'none', ...
    'MaxFunctionEvaluations', 250, ...
    'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-10);

try
    [z, resnorm, ~, exitflag] = lsqnonlin(fun, z0, lb, ub, opts);

    if exitflag <= 0 || sqrt(resnorm) > 1e-5
        sol = [];
    else
        sol = z;
    end
catch
    sol = [];
end

end


function Gc = cleaner_glucose_equilibrium(Yhc, D, P)

global EQ_EVALS
EQ_EVALS = EQ_EVALS + 1;

Dc = cleaner_threshold_Dc(Yhc, P);

if D >= Dc
    Gc = P.Gin;
    return;
end

fun = @(G) (1 - Yhc) * fc(G, 0.0, P) - P.kdeg - D;

opts = optimoptions('lsqnonlin', ...
    'Display', 'none', ...
    'MaxFunctionEvaluations', 150, ...
    'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-10);

try
    [Gsol, resnorm, ~, exitflag] = lsqnonlin(fun, P.Gin / 2, 1e-9, P.Gin, opts);

    if exitflag <= 0 || sqrt(resnorm) > 1e-5
        Gc = [];
    else
        Gc = Gsol;
    end
catch
    Gc = [];
end

end


function eq = coexistence_equilibrium(Yhp, Yhc, D, P)

global EQ_EVALS
EQ_EVALS = EQ_EVALS + 1;

if ~is_feasible([Yhp, Yhc, D], P)
    eq = [];
    return;
end

fun = @(z) [
    (1 - Yhp) * fp(z(3), z(4), P) - P.kdeg - D;
    (1 - Yhc) * fc(z(3), z(4), P) - P.kdeg - D;
    D * (P.Gin - z(3)) - rgp(z(3), z(4), P) * z(1) - rgc(z(3), z(4), P) * z(2);
    -D * z(4) + ...
        (rover_p(z(3), z(4), P) - raup_p(z(3), z(4), P)) * z(1) - ...
        raup_c(z(3), z(4), P) * z(2)
];

z0 = [1.0, 1.0, P.Gin / 2, 0.5];
lb = [1e-9, 1e-9, 1e-9, 0.0];
ub = [1000, 1000, P.Gin, 1000];

opts = optimoptions('lsqnonlin', ...
    'Display', 'none', ...
    'MaxFunctionEvaluations', 400, ...
    'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-10);

try
    [z, resnorm, ~, exitflag] = lsqnonlin(fun, z0, lb, ub, opts);

    if exitflag <= 0 || sqrt(resnorm) > 1e-5
        eq = [];
    else
        eq = z;
    end
catch
    eq = [];
end

end


function ev = evaluate_solution(x, P)

Yhp = x(1);
Yhc = x(2);
D = x(3);

eq = coexistence_equilibrium(Yhp, Yhc, D, P);

if isempty(eq)
    ev = [];
    return;
end

Bp = eq(1);
Bc = eq(2);
G = eq(3);
A = eq(4);

Hp = (Yhp / (1 - Yhp)) * Bp;

if P.CLEANER_PRODUCES_PROTEIN
    Hc = (Yhc / (1 - Yhc)) * Bc;
else
    Hc = 0.0;
end

H_total = Hp + Hc;

productivity = D * H_total;
process_yield = H_total / P.Gin;

if ~isfinite(productivity) || ~isfinite(process_yield) || ...
   productivity <= 0 || process_yield <= 0
    ev = [];
    return;
end

ev.productivity = productivity;
ev.process_yield = process_yield;
ev.Bp = Bp;
ev.Bc = Bc;
ev.G = G;
ev.A = A;
ev.Hp = Hp;
ev.Hc = Hc;
ev.H_total = H_total;

end



% TASAS BIOLÓGICAS

function y = rgp(G, A, P)

y = P.kg * G / (G + P.Kg) * P.theta_a / (A + P.theta_a);

end


function y = rgc(G, A, P)

y = P.beta * rgp(G, A, P);

end


function y = downreg(v, P)

phi = P.g / P.l;
y = max(0.0, (1.0 + phi) * P.g / (P.g + v) - phi);

end


function y = rover_p(G, A, P)

y = P.kover * max(0.0, rgp(G, A, P) - P.l);

end


function y = raup_p(G, A, P)

if A <= 0
    y = 0.0;
else
    y = P.ka * A / (A + P.Ka) * downreg(rgp(G, A, P), P);
end

end


function y = raup_c(G, A, P)

if A <= 0
    y = 0.0;
else
    y = ...
        P.ka * A / (A + P.Ka) * downreg(rgc(G, A, P), P) + ...
        P.kAcs * A / (A + P.KAcs);
end

end


function y = fp(G, A, P)

y = P.Yg * rgp(G, A, P) - P.Ya * rover_p(G, A, P) + P.Ya * raup_p(G, A, P);

end


function y = fc(G, A, P)

y = P.Yg * rgc(G, A, P) + P.Ya * raup_c(G, A, P);

end



% MÉTRICAS MULTIOBJETIVO

function idx = nondominated_indices_max(points)

if isempty(points)
    idx = [];
    return;
end

valid = all(isfinite(points), 2) & all(points > 0, 2);
points_valid = points(valid, :);
original_idx = find(valid);

if isempty(points_valid)
    idx = [];
    return;
end

rounded = round(points_valid * 1e8) / 1e8;
[points_unique, ia] = unique(rounded, 'rows', 'stable');

n = size(points_unique, 1);
dominated = false(n, 1);

for i = 1:n
    for j = 1:n
        if i == j
            continue;
        end

        if all(points_unique(j, :) >= points_unique(i, :)) && ...
           any(points_unique(j, :) > points_unique(i, :))
            dominated(i) = true;
            break;
        end
    end
end

idx_unique_nd = ia(~dominated);
idx = original_idx(idx_unique_nd);
idx = unique(idx, 'stable');

end


function hv = hypervolume_2d_max(points, ref)

if isempty(points)
    hv = 0.0;
    return;
end

pts = points(all(isfinite(points),2), :);
pts = pts(pts(:,1) > ref(1) & pts(:,2) > ref(2), :);

if isempty(pts)
    hv = 0.0;
    return;
end

nd_idx = nondominated_indices_max(pts);
pts = pts(nd_idx, :);

[~, idx] = sort(pts(:, 1), 'ascend');
pts = pts(idx, :);

hv = 0.0;
previous_x = ref(1);

for i = 1:size(pts, 1)
    x = pts(i, 1);
    y = pts(i, 2);

    width = x - previous_x;
    height = y - ref(2);

    if width > 0 && height > 0
        hv = hv + width * height;
        previous_x = x;
    end
end

end


function s = spacing_metric_python_style(points)

if isempty(points) || size(points, 1) < 3
    s = NaN;
    return;
end

pts = sortrows(points, 1);
distances = sqrt(sum(diff(pts, 1, 1).^2, 2));
s = std(distances);

end


% IGD+ y Epsilon indicador (minimización)

function val = igd_plus(A, Z)
    % A: aproximación (n x m), Z: referencia (k x m), ambos en minimización
    if isempty(Z) || isempty(A)
        val = inf;
        return;
    end
    n = size(A,1);
    k = size(Z,1);
    dist = zeros(k,1);
    for j = 1:k
        diff = Z(j,:) - A;               % n x m
        diff = max(0, diff);
        d2 = sum(diff.^2, 2);
        dist(j) = sqrt(min(d2));
    end
    val = mean(dist);
end

function eps = epsilon_indicator(A, Z)
    % A: n x m, Z: k x m, en minimización
    if isempty(Z) || isempty(A)
        eps = inf;
        return;
    end
    k = size(Z,1);
    max_eps = zeros(k,1);
    for j = 1:k
        delta = A - Z(j,:);              % n x m
        max_delta = max(delta, [], 2);    % máximo por fila
        max_eps(j) = min(max_delta);      % mejor a para z
    end
    eps = max(max_eps);
end

function ref = compute_reference_front(fronts)
    all_points = [];
    for i = 1:length(fronts)
        if ~isempty(fronts(i).points_nd)
            all_points = [all_points; fronts(i).points_nd];
        end
    end
    if isempty(all_points)
        ref = [];
        return;
    end
    nd_idx = nondominated_indices_max(all_points);
    ref = all_points(nd_idx, :);
end


% ESTADÍSTICA

function summary = summarize_metrics(metrics)

algorithms = unique(metrics.algorithm);

alg_col = {};
runs_col = [];

HV_mean = [];
HV_std = [];
HV_median = [];
HV_best = [];
HV_worst = [];

Spacing_mean = [];
Spacing_std = [];
Spacing_median = [];
Spacing_best = [];
Spacing_worst = [];

ND_mean = [];
ND_std = [];

time_mean = [];
time_std = [];

eval_mean = [];
eval_std = [];

obj_mean = [];
con_mean = [];
eq_mean = [];

successful_mean = [];

IGD_plus_mean = [];
IGD_plus_std = [];
Epsilon_mean = [];
Epsilon_std = [];

for i = 1:length(algorithms)
    alg = algorithms{i};
    sub = metrics(strcmp(metrics.algorithm, alg), :);

    alg_col{end+1,1} = alg;
    runs_col(end+1,1) = height(sub);

    HV_mean(end+1,1) = mean(sub.HV, 'omitnan');
    HV_std(end+1,1) = std(sub.HV, 'omitnan');
    HV_median(end+1,1) = median(sub.HV, 'omitnan');
    HV_best(end+1,1) = max(sub.HV);
    HV_worst(end+1,1) = min(sub.HV);

    Spacing_mean(end+1,1) = mean(sub.Spacing, 'omitnan');
    Spacing_std(end+1,1) = std(sub.Spacing, 'omitnan');
    Spacing_median(end+1,1) = median(sub.Spacing, 'omitnan');
    Spacing_best(end+1,1) = min(sub.Spacing);
    Spacing_worst(end+1,1) = max(sub.Spacing);

    ND_mean(end+1,1) = mean(sub.n_non_dominated, 'omitnan');
    ND_std(end+1,1) = std(sub.n_non_dominated, 'omitnan');

    time_mean(end+1,1) = mean(sub.time_seconds, 'omitnan');
    time_std(end+1,1) = std(sub.time_seconds, 'omitnan');

    eval_mean(end+1,1) = mean(sub.n_function_evaluations, 'omitnan');
    eval_std(end+1,1) = std(sub.n_function_evaluations, 'omitnan');

    obj_mean(end+1,1) = mean(sub.n_objective_function_calls, 'omitnan');
    con_mean(end+1,1) = mean(sub.n_constraint_function_calls, 'omitnan');
    eq_mean(end+1,1) = mean(sub.n_equilibrium_solver_calls, 'omitnan');

    successful_mean(end+1,1) = mean(sub.n_successful_weight_solutions, 'omitnan');

    IGD_plus_mean(end+1,1) = mean(sub.IGD_plus, 'omitnan');
    IGD_plus_std(end+1,1) = std(sub.IGD_plus, 'omitnan');
    Epsilon_mean(end+1,1) = mean(sub.Epsilon, 'omitnan');
    Epsilon_std(end+1,1) = std(sub.Epsilon, 'omitnan');
end

summary = table( ...
    alg_col, runs_col, ...
    HV_mean, HV_std, HV_median, HV_best, HV_worst, ...
    Spacing_mean, Spacing_std, Spacing_median, Spacing_best, Spacing_worst, ...
    ND_mean, ND_std, ...
    time_mean, time_std, ...
    eval_mean, eval_std, ...
    obj_mean, con_mean, eq_mean, successful_mean, ...
    IGD_plus_mean, IGD_plus_std, Epsilon_mean, Epsilon_std, ...
    'VariableNames', { ...
        'algorithm','runs', ...
        'HV_mean','HV_std','HV_median','HV_best','HV_worst', ...
        'Spacing_mean','Spacing_std','Spacing_median','Spacing_best','Spacing_worst', ...
        'ND_mean','ND_std', ...
        'time_mean','time_std', ...
        'eval_mean','eval_std', ...
        'objective_calls_mean','constraint_calls_mean','equilibrium_calls_mean', ...
        'successful_weight_solutions_mean', ...
        'IGD_plus_mean','IGD_plus_std','Epsilon_mean','Epsilon_std' ...
    } ...
);

end


function result = wilcoxon_analysis(metrics)

algorithms = unique(metrics.algorithm);

rows_metric = {};
rows_comparison = {};
rows_stat = [];
rows_p = [];
rows_status = {};

test_metrics = {'HV', 'Spacing', 'n_non_dominated', 'time_seconds', ...
                'n_function_evaluations', 'IGD_plus', 'Epsilon'};

if length(algorithms) < 2
    result = table({'NA'}, {'NA'}, NaN, NaN, {'Menos de dos algoritmos'}, ...
        'VariableNames', {'metric','comparison','statistic','p_value','status'});
    return;
end

ref_alg = algorithms{1};

for m = 1:length(test_metrics)
    metric = test_metrics{m};

    for a = 1:length(algorithms)
        alg = algorithms{a};

        if strcmp(alg, ref_alg)
            continue;
        end

        ref_sub = metrics(strcmp(metrics.algorithm, ref_alg), {'seed', metric});
        alg_sub = metrics(strcmp(metrics.algorithm, alg), {'seed', metric});

        merged = innerjoin(ref_sub, alg_sub, 'Keys', 'seed');

        if height(merged) < 2
            stat = NaN;
            pval = NaN;
            status = 'Datos insuficientes';
        else
            x = merged{:,2};
            y = merged{:,3};

            try
                [pval, ~, stats] = signrank(x, y);
                stat = stats.signedrank;
                status = 'OK';
            catch ME
                stat = NaN;
                pval = NaN;
                status = ME.message;
            end
        end

        rows_metric{end+1,1} = metric;
        rows_comparison{end+1,1} = [ref_alg ' vs ' alg];
        rows_stat(end+1,1) = stat;
        rows_p(end+1,1) = pval;
        rows_status{end+1,1} = status;
    end
end

result = table(rows_metric, rows_comparison, rows_stat, rows_p, rows_status, ...
    'VariableNames', {'metric','comparison','statistic','p_value','status'});

end


function result = holm_correction(wilcoxon_results)

if isempty(wilcoxon_results)
    result = table();
    return;
end

rows_metric = {};
rows_comparison = {};
rows_p = [];
rows_threshold = [];
rows_adjusted = [];
rows_reject = [];
rows_status = {};

metrics_unique = unique(wilcoxon_results.metric);

for m = 1:length(metrics_unique)
    metric_name = metrics_unique{m};
    sub = wilcoxon_results(strcmp(wilcoxon_results.metric, metric_name), :);

    valid = ~isnan(sub.p_value);
    sub_valid = sub(valid,:);

    if isempty(sub_valid)
        continue;
    end

    [~, order] = sort(sub_valid.p_value, 'ascend');
    sub_valid = sub_valid(order,:);

    n_tests = height(sub_valid);

    for i = 1:n_tests
        denominator = n_tests - i + 1;
        threshold = 0.05 / denominator;
        adjusted_p = min(sub_valid.p_value(i) * denominator, 1.0);
        reject = sub_valid.p_value(i) <= threshold;

        rows_metric{end+1,1} = metric_name;
        rows_comparison{end+1,1} = sub_valid.comparison{i};
        rows_p(end+1,1) = sub_valid.p_value(i);
        rows_threshold(end+1,1) = threshold;
        rows_adjusted(end+1,1) = adjusted_p;
        rows_reject(end+1,1) = reject;
        rows_status{end+1,1} = 'OK';
    end
end

result = table(rows_metric, rows_comparison, rows_p, rows_threshold, ...
    rows_adjusted, rows_reject, rows_status, ...
    'VariableNames', {'metric','comparison','p_value', ...
    'holm_threshold','holm_adjusted_p','holm_reject_0_05','status'});

end


function result = friedman_analysis(metrics)

algorithms = unique(metrics.algorithm);
test_metrics = {'HV', 'Spacing', 'n_non_dominated', 'time_seconds', ...
                'n_function_evaluations', 'IGD_plus', 'Epsilon'};

rows_metric = {};
rows_stat = [];
rows_p = [];
rows_status = {};

for m = 1:length(test_metrics)
    metric = test_metrics{m};

    if length(algorithms) < 3
        rows_metric{end+1,1} = metric;
        rows_stat(end+1,1) = NaN;
        rows_p(end+1,1) = NaN;
        rows_status{end+1,1} = 'Friedman requiere al menos 3 algoritmos';
        continue;
    end

    seeds = unique(metrics.seed);
    data = [];

    for s = 1:length(seeds)
        seed = seeds(s);
        row = [];
        complete = true;

        for a = 1:length(algorithms)
            alg = algorithms{a};
            sub = metrics(strcmp(metrics.algorithm, alg) & metrics.seed == seed, :);

            if isempty(sub)
                complete = false;
                break;
            end

            row = [row, sub.(metric)(1)];
        end

        if complete
            data = [data; row];
        end
    end

    if size(data,1) < 2
        stat = NaN;
        pval = NaN;
        status = 'Datos insuficientes';
    else
        try
            [pval, tbl] = friedman(data, 1, 'off');
            stat = tbl{2,5};
            status = 'OK';
        catch ME
            stat = NaN;
            pval = NaN;
            status = ME.message;
        end
    end

    rows_metric{end+1,1} = metric;
    rows_stat(end+1,1) = stat;
    rows_p(end+1,1) = pval;
    rows_status{end+1,1} = status;
end

result = table(rows_metric, rows_stat, rows_p, rows_status, ...
    'VariableNames', {'metric','statistic','p_value','status'});

end


function plot_fronts(all_fronts)

figure;
hold on;
grid on;

for i = 1:length(all_fronts)
    pts = all_fronts(i).points_nd;

    if isempty(pts)
        continue;
    end

    scatter(pts(:,1), pts(:,2), 18, 'filled');
end

xlabel('Productividad D \cdot H^*');
ylabel('Rendimiento H^*/G_{in}');
title('Frentes no dominados fmincon - 30 corridas');
hold off;

end
