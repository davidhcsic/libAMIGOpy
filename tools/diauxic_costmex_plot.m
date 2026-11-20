% diauxic_costmex_plot.m — diauxic shift model, 10 glucose concentrations,
% simulated with the fully compiled costMex MEX and plotted.
%
% Classic diauxie model (glucose + ethanol as secondary carbon source):
%   Glucose (S) is consumed first via the preferred uptake (u1). While S > 0,
%   the repressor R stays OFF (R ~ 1), blocking ethanol uptake (u2 = 0).
%   Ethanol (E) accumulates as a by-product of glucose metabolism.
%   When glucose runs out, R slowly degrades, u2 turns on, and cells grow
%   on ethanol — producing the characteristic diauxic lag.
%
%   u1 = umax1 * S/(Ks1+S) * (1-R)
%   u2 = umax2 * E/(Ks2+E) * R
%   dS = -u1/Y1 * X
%   dE = (q_prod*u1 - u2/Y2) * X
%   dX = (u1 + u2) * X
%   dR = (k_on*(1-R)*S/(Ks1+S) - k_off*R)
%
% 10 experiments = 10 initial glucose concentrations S0 (as initial conditions).
% Requires MATLAB + the compiled costMex stack (basic_amigo).

this_dir = fileparts(mfilename('fullpath'));
basic_dir = fullfile(this_dir, '..', '..', 'basic_amigo');
addpath(basic_dir);
addpath(fullfile(basic_dir, 'core'));
addpath(fullfile(basic_dir, 'libAMIGO'));

clear mex;

%% Problem definition ---------------------------------------------------------
inputs.pathd.results_folder = 'py_diauxic';
inputs.pathd.short_name     = 'py_diauxic';

inputs.model.input_model_type = 'charmodelC';
inputs.model.exe_type         = 'costMex';

inputs.model.n_st     = 4;
inputs.model.st_names = char({'S'; 'E'; 'X'; 'R'});
inputs.model.eqns = char({
    'u1=umax1*S/(Ks1+S)', ...
    'u2=umax2*E/(Ks2+E)*(1-R)', ...
    'dS=-u1/Y1*X', ...
    'dE=(qprod*u1-u2/Y2)*X', ...
    'dX=(u1+u2)*X', ...
    'dR=-k_on*R*S/(Ks1+S)+k_off*(1-R)'});

inputs.model.n_par      = 9;
inputs.model.par_names  = char({'umax1'; 'umax2'; 'Ks1'; 'Ks2'; ...
                                'Y1'; 'Y2'; 'k_on'; 'k_off'; 'qprod'});
inputs.model.par        = [0.30; 0.10; 0.05; 0.30; 0.5; 0.4; 2.0; 0.05; 0.06];

inputs.ivpsol.rtol = 1e-7;
inputs.ivpsol.atol = 1e-8;
inputs.ivpsol.max_step_size  = 10;
inputs.ivpsol.ivp_maxnumsteps = 100000;
inputs.ivpsol.nthreads = 8;

inputs.exps.data_type  = 'real';
inputs.exps.noise_type = 'homo_var';
inputs.PEsol.PEcost_type = 'lsq';

inputs.PEsol.id_global_theta  = char({'umax1'; 'umax2'});
inputs.PEsol.global_theta_min = [0.01 0.01];
inputs.PEsol.global_theta_max = [10 10];

%% 10 experiments: glucose concentration sweep --------------------------------
S0_values = [0.2 0.5 1 2 4 8 12 16 20 30];
n_exp = numel(S0_values);

for iexp = 1:n_exp
    t = linspace(0, 60, 300);

    inputs.exps.n_exp = n_exp;

    inputs.exps.obs{iexp}      = char({'Xobs=X', 'Sobs=S', 'Eobs=E'});
    inputs.exps.obs_names{iexp} = char({'Xobs'; 'Sobs'; 'Eobs'});
    inputs.exps.n_obs{iexp}    = 3;

    inputs.exps.exp_data{iexp}  = zeros(numel(t), 3);
    inputs.exps.error_data{iexp}= ones(numel(t), 3);

    inputs.exps.t_s{iexp} = t;
    inputs.exps.n_s{iexp} = numel(t);
    inputs.exps.t_f{iexp} = t(end);
    inputs.exps.t_in{iexp} = 0;
    inputs.exps.exp_y0{iexp} = [S0_values(iexp); 0; 0.05; 0];  % S0, E0=0, X0, R=0 (repressed)
end

%% Prepare (generate C + compile costMex) and simulate ------------------------
[inputs, privstruct] = mini_AMIGO_prep(inputs);
results = mini_AMIGO_sim(inputs, privstruct);

assert(all([results.sim.success{:}]), 'Some experiments failed to integrate.');

%% Plot -----------------------------------------------------------------------
cmap = lines(n_exp);
figure('Color', 'w', 'Position', [80 80 1500 450]);

for ipanel = 1:3
    subplot(1, 3, ipanel); hold on; grid on; box on;
end
lg = cell(1, n_exp);
for iexp = 1:n_exp
    t  = results.sim.tsim{iexp};
    ob = results.sim.obs{iexp};
    subplot(1, 3, 1); plot(t, ob(:,1), 'Color', cmap(iexp,:), 'LineWidth', 1.4);
    subplot(1, 3, 2); plot(t, ob(:,2), 'Color', cmap(iexp,:), 'LineWidth', 1.4);
    subplot(1, 3, 3); plot(t, ob(:,3), 'Color', cmap(iexp,:), 'LineWidth', 1.4);
    lg{iexp} = sprintf('S0 = %g', S0_values(iexp));
end
subplot(1,3,1); xlabel('Time [h]'); ylabel('Biomass X [g/L]');
title('Diauxic growth'); legend(lg, 'Location', 'southeast');
subplot(1,3,2); xlabel('Time [h]'); ylabel('Glucose S [g/L]');
title('Glucose consumption'); legend(lg, 'Location', 'northeast');
subplot(1,3,3); xlabel('Time [h]'); ylabel('Ethanol E [g/L]');
title('Ethanol (by-product then consumed)'); legend(lg, 'Location', 'northwest');
sgtitle('Diauxic shift model — 10 glucose concentrations (compiled costMex MEX)');

out_png = fullfile(this_dir, 'diauxic_costmex_plot.png');
saveas(gcf, out_png);
fprintf('Saved: %s\n', out_png);
