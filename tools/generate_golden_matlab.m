function generate_golden_matlab(case_name)
% generate_golden_matlab: run the compiled costMex MEX and dump one golden JSON.
%
% Golden files are the reference for the Python port (CONSTITUTION Art. III).
% One case per MATLAB session (MEX reload in-session is unstable):
%   export TMPDIR=/tmp
%   matlab -batch "cd('<this_dir>'); generate_golden_matlab('mini_nominal')"
% cases: mini_nominal | mini_theta_alt | mini_ramp_theta

this_dir = fileparts(mfilename('fullpath'));
root     = fileparts(this_dir);                      % python/
basic    = fullfile(root, '..', 'basic_amigo');

addpath(basic);
addpath(fullfile(basic, 'core'));
addpath(fullfile(basic, 'libAMIGO'));

golden_dir = fullfile(this_dir, 'golden');
if ~isdir(golden_dir), mkdir(golden_dir); end

clear mex;

switch case_name
    case {'mini_nominal', 'mini_theta_alt'}
        inputs = local_mini_inputs();
        [inputs, privstruct] = mini_AMIGO_prep(inputs);
        if strcmp(case_name, 'mini_theta_alt')
            theta = [0.25; 0.05];
        else
            theta = [];
        end
        [results, inputs] = mini_AMIGO_sim(inputs, privstruct, theta);
        feval(inputs.model.mexfunction, 'cost_LSQ');  % -> 'outputs' in caller
        cost_out = outputs;

    case 'mini_ramp_theta'
        inputs = local_mini_inputs();
        inputs.pathd.short_name = 'mini_golden_ramp';
        inputs.pathd.results_folder = 'mini_golden_ramp';
        for iexp = 1:2
            t = linspace(0, 30, 25);
            inputs.exps.t_s{iexp} = t;
            inputs.exps.n_s{iexp} = numel(t);
            inputs.exps.t_f{iexp} = t(end);
            inputs.exps.u_interp{iexp} = 'linear';
            inputs.exps.n_linear{iexp} = 2;
            inputs.exps.t_con{iexp} = [0, 30];
            inputs.exps.u{iexp} = [0.2, 1.0 + 0.5*iexp];
        end
        theta = [0.08; 0.03];
        [inputs, privstruct] = mini_AMIGO_prep(inputs);
        [results, inputs] = mini_AMIGO_sim(inputs, privstruct, theta);
        feval(inputs.model.mexfunction, 'cost_LSQ');  % -> 'outputs' in caller
        cost_out = outputs;

    otherwise
        error('unknown golden case: %s', case_name);
end

if isempty(theta)
    theta = inputs.model.par(inputs.PEsol.index_global_theta);
end

local_dump(fullfile(golden_dir, [case_name '.json']), inputs, results, theta, cost_out);
fprintf('generate_golden_matlab: wrote %s\n', fullfile(golden_dir, [case_name '.json']));
end

function inputs = local_mini_inputs()
inputs.pathd.results_folder = 'mini_amigo';
inputs.pathd.short_name     = 'mini_demo';
inputs.model.input_model_type = 'charmodelC';
inputs.model.exe_type         = 'costMex';
inputs.model.n_st     = 1;
inputs.model.st_names = 'X';
inputs.model.eqns     = 'dX=k1*S0*X-k2*X';
inputs.model.n_par      = 2;
inputs.model.par_names  = char({'k1'; 'k2'});
inputs.model.par        = [0.1; 0.02];
inputs.model.n_stimulus     = 1;
inputs.model.stimulus_names = 'S0';
inputs.ivpsol.rtol = 1e-5;
inputs.ivpsol.atol = 1e-6;
inputs.ivpsol.max_step_size  = 100;
inputs.ivpsol.ivp_maxnumsteps = 10000;
inputs.ivpsol.nthreads = 4;
inputs.exps.data_type  = 'real';
inputs.exps.noise_type = 'homo_var';
inputs.PEsol.PEcost_type = 'lsq';
inputs.PEsol.id_global_theta  = char({'k1'; 'k2'});
inputs.PEsol.global_theta_min = [0.001; 0.0001]';
inputs.PEsol.global_theta_max = [10;    10]';
n_exp = 2;
for iexp = 1:n_exp
    t = linspace(0, 60, 40);
    inputs.exps.n_exp = n_exp;
    inputs.exps.obs{iexp}      = char({'Xobs=X', 'X2obs=2*X'});
    inputs.exps.obs_names{iexp} = char({'Xobs'; 'X2obs'});
    inputs.exps.n_obs{iexp}    = 2;
    inputs.exps.exp_data{iexp}  = zeros(numel(t), 2);
    inputs.exps.error_data{iexp}= ones(numel(t), 2);
    inputs.exps.t_s{iexp} = t;
    inputs.exps.n_s{iexp} = numel(t);
    inputs.exps.t_f{iexp} = t(end);
    inputs.exps.t_in{iexp} = 0;
    inputs.exps.exp_y0{iexp} = 1;
    inputs.exps.u_interp{iexp} = 'step';
    inputs.exps.t_con{iexp}    = [0, t(2:end-1)-0.5*mean(diff(t)), t(end)];
    inputs.exps.n_steps{iexp}  = numel(t) - 1;
    inputs.exps.u{iexp}        = repmat(0.5 + 0.5*iexp, 1, numel(t)-1);
end
end

function local_dump(fname, inputs, results, theta, cost_out)
fid = fopen(fname, 'w');
fprintf(fid, '{\n');
fprintf(fid, '"description": "golden reference from MATLAB costMex MEX",\n');
fprintf(fid, '"tolerance_rel": 1e-6,\n');
fprintf(fid, '"theta": [%s],\n', strjoin(arrayfun(@local_str, theta(:)', 'UniformOutput', false), ', '));
fprintf(fid, '"par": [%s],\n', strjoin(arrayfun(@local_str, inputs.model.par(:)', 'UniformOutput', false), ', '));
fprintf(fid, '"n_exp": %d,\n', inputs.exps.n_exp);
fprintf(fid, '"exps": [\n');
for iexp = 1:inputs.exps.n_exp
    ts = results.sim.tsim{iexp}(:);
    ob = results.sim.obs{iexp};
    sm = results.sim.states{iexp};
    fprintf(fid, '{\n');
    fprintf(fid, '"t_s": [%s],\n', strjoin(arrayfun(@local_str, ts', 'UniformOutput', false), ', '));
    fprintf(fid, '"obs": [\n');
    for j = 1:size(ob, 2)
        fprintf(fid, '[%s]%s\n', strjoin(arrayfun(@local_str, ob(:,j)', 'UniformOutput', false), ', '), local_comma(j, size(ob,2)));
    end
    fprintf(fid, '],\n');
    fprintf(fid, '"states": [\n');
    for j = 1:size(sm, 2)
        fprintf(fid, '[%s]%s\n', strjoin(arrayfun(@local_str, sm(:,j)', 'UniformOutput', false), ', '), local_comma(j, size(sm,2)));
    end
    fprintf(fid, '],\n');
    fprintf(fid, '"success": %s\n', local_bool(results.sim.success{iexp}));
    if iexp < inputs.exps.n_exp, fprintf(fid, '},\n'); else, fprintf(fid, '}\n'); end
end
fprintf(fid, '],\n');
fprintf(fid, '"cost_f": %s,\n', local_str(cost_out.f));
fprintf(fid, '"cost_w_res": [%s]\n', strjoin(arrayfun(@local_str, cost_out.w_res(:)', 'UniformOutput', false), ', '));
fprintf(fid, '}\n');
fclose(fid);
end

function s = local_str(v)
% 17 significant digits so the JSON round-trips as the exact double.
s = strtrim(sprintf('%.17g', v));
end

function s = local_comma(j, n)
if j < n, s = ','; else, s = ''; end
end

function s = local_bool(b)
if b, s = 'true'; else, s = 'false'; end
end
