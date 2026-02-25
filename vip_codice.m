%% =======================================================================
%  Caso Virginia Investment Partners
%  File dati: dataset_rendimenti.csv
% ========================================================================

clc; clear; close all;

%% 0. Caricamento dati

T = readtable('dataset_rendimenti.csv');   % Il file deve essere nel path

disp('Nomi colonne nel file:');
disp(T.Properties.VariableNames);

% Estraggo i rendimenti (escludo la colonna Date)
R = T{:, 2:end};   % matrice T x 4: [S&P, MSCI, LehmanAgg, IBM]

% Assegno per chiarezza
r_sp   = R(:,1);   % S&P 500
r_msci = R(:,2);   % MSCI World ex US
r_agg  = R(:,3);   % Lehman Brothers Aggregate Bond Index
r_ibm  = R(:,4);   % IBM

%% 1. Mean e standard deviation (mensile e annualizzata) per TUTTI gli asset
% ------------------------------------------------------------------------

assetNames = {'S&P 500', 'MSCI World ex US', 'Lehman Agg Bond', 'IBM'};

% Statistiche mensili
mu_month  = mean(R, 1);          % 1 x 4 (media mensile)
sigma_mon = std(R, 0, 1);        % 1 x 4 (dev.std mensile, sample)

% Annualizzazione
mu_ann    = (1 + mu_month).^12 - 1;   % da mensile ad annuale
sigma_ann = sigma_mon * sqrt(12);     % std annua

% Stampa tabella in percentuale
fprintf('================ STATISTICHE ASSET (%%) ================\n');
fprintf('%-20s  %12s  %12s  %12s  %12s\n', ...
        'Asset', 'Mean mon', 'Std mon', 'Mean ann', 'Std ann');
fprintf('%s\n', repmat('-',1,80));

for i = 1:numel(assetNames)
    fprintf('%-20s  %11.2f%%  %11.2f%%  %11.2f%%  %11.2f%%\n', ...
        assetNames{i}, ...
        mu_month(i)*100, sigma_mon(i)*100, ...
        mu_ann(i)*100,   sigma_ann(i)*100);
end

fprintf('%s\n\n', repmat('=',1,80));

% Estraggo IBM per comodo
mu_ibm_month  = mu_month(4);
sigma_ibm_mon = sigma_mon(4);
mu_ibm_ann    = mu_ann(4);
sigma_ibm_ann = sigma_ann(4);

%% 2. Portafoglio equally-weighted (S&P, MSCI, LehmanAgg)
% ------------------------------------------------------------------------

w_eq = [1/3; 1/3; 1/3];     % pesi equal-weighted per le 3 asset class
R_assets = [r_sp, r_msci, r_agg];   % solo le tre asset class

% Rendimenti mensili del portafoglio equally weighted
r_p_eq_month = R_assets * w_eq;

% Statistiche mensili
mu_p_eq_month  = mean(r_p_eq_month);
sigma_p_eq_mon = std(r_p_eq_month);

% Annualizzazione
mu_p_eq_ann    = (1 + mu_p_eq_month).^12 - 1;
sigma_p_eq_ann = sigma_p_eq_mon * sqrt(12);

fprintf('--- Punto 2: Portafoglio equally-weighted (S&P, MSCI, LehmanAgg) ---\n');
fprintf('Media mensile:         %.2f%%\n', mu_p_eq_month*100);
fprintf('Dev.std mensile:       %.2f%%\n', sigma_p_eq_mon*100);
fprintf('Media annualizzata:    %.2f%%\n', mu_p_eq_ann*100);
fprintf('Dev.std annualizzata:  %.2f%%\n\n', sigma_p_eq_ann*100);

%% 3. Mean-variance optimizer con target sigma = 10%% (annualizzato)
% ------------------------------------------------------------------------

% Statistiche delle 3 asset class (annuali)
mu_assets_month = mean(R_assets);                % 1 x 3
mu_assets_ann   = (1 + mu_assets_month).^12 - 1; % 1 x 3
Sigma_month     = cov(R_assets);                 % 3 x 3
Sigma_ann       = Sigma_month * 12;              % matrice cov annua

% Target di volatilità (ANNUALE)
sigma_target_10 = 0.10;   % 10%

% Funzione obiettivo: max mu*w  <=> min -mu*w
% w(:) forza il vettore a colonna (3x1) così il prodotto è ben definito
f_obj = @(w) -mu_assets_ann * w(:);

% Vincoli lineari: somma pesi = 1, nessun altro vincolo lineare
Aeq = [1 1 1];
beq = 1;
A   = [];
b   = [];

% Limiti sui pesi (no short selling)
lb = zeros(3,1);
ub = ones(3,1);

% Vincolo non lineare: std_portafoglio = sigma_target
nonlcon_10 = @(w) port_sigma_constraint(w, Sigma_ann, sigma_target_10);

% Punto iniziale
w0 = ones(3,1) / 3;

options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'sqp');

[w_opt_10, fval_10, exitflag_10, output_10] = fmincon( ...
    f_obj, w0, A, b, Aeq, beq, lb, ub, nonlcon_10, options);

w_opt_10 = w_opt_10(:);   % forzo colonna

% Rendimento e rischio del portafoglio ottimale
mu_p_10    = mu_assets_ann * w_opt_10;
sigma_p_10 = sqrt(w_opt_10' * Sigma_ann * w_opt_10);

fprintf('--- Punto 3: Portafoglio ottimale con target sigma = 10%% ---\n');
fprintf('Pesi ottimali (S&P, MSCI, LehmanAgg): [%.3f  %.3f  %.3f]\n', w_opt_10);
fprintf('Rendimento atteso annuo portafoglio:  %.2f%%\n', mu_p_10*100);
fprintf('Volatilità annua portafoglio:         %.2f%%\n\n', sigma_p_10*100);

%% 4. Ripetizione per target sigma = 2%%, 6%%, 10%%, 14%%, 20%%
% ------------------------------------------------------------------------

sigma_targets = [0.02 0.06 0.10 0.14 0.20];  % annuali
nT = numel(sigma_targets);

W_opt   = zeros(3, nT);   % pesi per ciascun target
mu_p    = zeros(1, nT);   % rendimenti portafogli
sigma_p = zeros(1, nT);   % volatilità portafogli

for i = 1:nT
    sig_t = sigma_targets(i);
    nonlcon_i = @(w) port_sigma_constraint(w, Sigma_ann, sig_t);
    
    [w_i, fval_i, exitflag_i] = fmincon( ...
        f_obj, w0, A, b, Aeq, beq, lb, ub, nonlcon_i, options);
    
    w_i = w_i(:);
    W_opt(:,i) = w_i;
    mu_p(i)    = mu_assets_ann * w_i;
    sigma_p(i) = sqrt(w_i' * Sigma_ann * w_i);
    
    fprintf('--- Punto 4: Target sigma = %.0f%%%% ---\n', sig_t*100);
    fprintf('Pesi ottimali [S&P, MSCI, LehmanAgg]: [%.3f  %.3f  %.3f]\n', w_i);
    fprintf('Rendimento atteso annuo: %.2f%%\n', mu_p(i)*100);
    fprintf('Volatilità annua:        %.2f%%\n\n', sigma_p(i)*100);
end

%% 5. Grafico rischio-rendimento (portafogli ottimizzati vs IBM)
% ------------------------------------------------------------------------

figure;
hold on; grid on; box on;

% Portafogli ottimizzati (per i vari target di sigma)
plot(sigma_p*100, mu_p*100, 'o-', 'LineWidth', 1.5, 'MarkerSize', 8);

% Punto IBM
plot(sigma_ibm_ann*100, mu_ibm_ann*100, ...
     'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Punto portafoglio equally weighted
plot(sigma_p_eq_ann*100, mu_p_eq_ann*100, ...
     'kd', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

xlabel('Volatilità annua (%)');
ylabel('Rendimento atteso annuo (%)');
title('Confronto rischio-rendimento: Portafogli ottimizzati vs IBM');
legend({'Portafogli target \sigma', 'IBM 100%', 'Portafoglio 1/3-1/3-1/3'}, ...
       'Location', 'best');

hold off;


%% =======================================================================
%  Funzione di vincolo non lineare per fmincon
% =======================================================================
function [c, ceq] = port_sigma_constraint(w, Sigma, sigma_target)
    % Forzo w a vettore colonna
    w = w(:);
    sigma_p = sqrt(w' * Sigma * w);
    c   = [];                       % nessun vincolo di tipo <=
    ceq = sigma_p - sigma_target;   % vincolo di uguaglianza
end
