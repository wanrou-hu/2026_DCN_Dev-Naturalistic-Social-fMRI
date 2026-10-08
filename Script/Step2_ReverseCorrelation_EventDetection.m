% ============================================================
% Reverse correlation analysis for social event detection
% ============================================================
% This script identifies ToM- and Pain-related events during
% naturalistic movie viewing based on adult network responses.
%
% Analysis workflow:
% 1. Extract subject-level network activity by temporally
%    z-scoring each ROI and averaging signals across ROIs.
% 2. Identify TRs showing significant positive group-level
%    network activation using one-tailed one-sample t-tests.
% 3. Detect events with at least two consecutive significant
%    TRs and extract their onset times and durations.
%
% Event detection: corrected p < 0.05; minimum 2 TRs.
% ============================================================

clc;clear;close all

%% Step 1: Paths and analysis parameters
project_dir = fileparts(fileparts(mfilename('fullpath')));
input_dir = fullfile(project_dir, 'data', 'ROIsignals', 'Adults');

event_types = {'TOM', 'Pain'};

TR = 2;                    % Repetition time (seconds)
n_TR = 150;                % Number of analyzed time points
p_threshold = 0.05;        % Uncorrected significance threshold
min_consecutive_tr = 2;    % Minimum event duration in TRs


%% Step 2: Extract subject-level network activity
for i_event = 1:numel(event_types)
    % Load adult ROI time series for the current network
    % Each file contains ROISignals (TR x ROI)
    network_dir = fullfile(input_dir, event_types{i_event});
    file_list = dir(fullfile(network_dir, '*.mat'));


    n_subjects = numel(file_list);
    Z_all = zeros(n_subjects, n_TR);

    for i = 1:n_subjects
        file_name = fullfile(file_list(i).folder, file_list(i).name);
        data = load(file_name, 'ROISignals');
        ROISignals = data.ROISignals;

        % Normalize each ROI across time within each participant
        z_ROISignals = zscore(ROISignals, 0, 1);

        % Average across ROIs to obtain one network time course
        Z_all(i,:) = mean(z_ROISignals, 2)';
    end


    %% Step 3: Identify significant network activation
    % Perform a right-tailed one-sample t-test against zero
    % across adult participants independently at each TR
    P = zeros(1, n_TR);

    for j = 1:n_TR
        [~, P(j)] = ttest(Z_all(:, j), 0, 'Tail', 'right');
    end

    % Mark TRs with significant positive network responses
    significant_TR = P < p_threshold;


    %% Step 4: Detect contiguous significant events
    % Identify the start and end of each consecutive run of significant TRs
    event_edges = diff([false, significant_TR, false]);

    event_onsets = find(event_edges == 1);
    event_ends = find(event_edges == -1);

    % Calculate event lengths as the number of included TRs
    event_durations = event_ends - event_onsets;

    % Retain events containing at least two consecutive TRs
    valid_events = event_durations >= min_consecutive_tr;

    onset_TR = event_onsets(valid_events);
    duration_TR = event_durations(valid_events);

    % Convert to time in seconds
    onset_sec = (onset_TR - 1) * TR;
    duration_sec = duration_TR * TR;
end