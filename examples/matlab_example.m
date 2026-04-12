% Insight137 EAP — MATLAB Integration Example
% Load results exported from Python library
%
% Step 1: In Python, run:
%   import insight137_eap as eap
%   profiles = [eap.compute_psi_from_sequence(eap.examples.human_keystrokes),
%               eap.compute_psi_from_sequence(eap.examples.bot_keystrokes)]
%   eap.to_matlab(profiles, 'eap_results.mat', labels=['human', 'bot'])
%
% Step 2: In MATLAB, run this script:

data = load('eap_results.mat');

% Display results
disp('Psi Profiles:');
disp(table(data.labels', data.psi_1', data.psi_2', data.psi_3', data.psi_4', ...
    'VariableNames', {'Label', 'Psi1', 'Psi2', 'Psi3', 'Psi4'}));

% Radar chart in MATLAB
theta = linspace(0, 2*pi, 5);
labels_ax = {'Psi1', 'Psi2', 'Psi3', 'Psi4', 'Psi1'};

figure('Name', 'EAP Psi Profile Comparison');
for i = 1:length(data.labels)
    vals = [data.psi_1(i), data.psi_2(i), data.psi_3(i), data.psi_4(i), data.psi_1(i)];
    polarplot(theta, vals, '-o', 'LineWidth', 2, 'DisplayName', data.labels{i});
    hold on;
end
legend('Location', 'best');
title('Entropy Attunement Profile');
