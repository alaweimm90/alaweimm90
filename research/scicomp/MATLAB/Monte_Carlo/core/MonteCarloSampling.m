function sampler = MonteCarloSampling()
% MonteCarloSampling - Monte Carlo sampling methods for Berkeley SciComp
%
% Provides comprehensive Monte Carlo sampling methods including:
% - Random number generation
% - Markov Chain Monte Carlo
% - Importance sampling
% - Metropolis-Hastings algorithm
%
% Author: UC Berkeley SciComp Team
% Copyright © 2025 Meshal Alawein — All rights reserved.
    sampler.uniformSample = @uniformSample;
    sampler.normalSample = @normalSample;
    sampler.metropolisHastings = @metropolisHastings;
    sampler.importanceSampling = @importanceSampling;
    sampler.mcmcSampling = @mcmcSampling;
    sampler.gibbsSampling = @gibbsSampling;
    sampler.monteCarloIntegration = @monteCarloIntegration;
end
function samples = uniformSample(n, a, b)
% Generate uniform random samples
%
% Args:
%   n: Number of samples
%   a: Lower bound (default 0)
%   b: Upper bound (default 1)
%
% Returns:
%   samples: Array of uniform random samples
    if nargin < 2, a = 0; end
    if nargin < 3, b = 1; end
    samples = a + (b - a) * rand(n, 1);
end
function samples = normalSample(n, mu, sigma)
% Generate normal random samples
%
% Args:
%   n: Number of samples
%   mu: Mean (default 0)
%   sigma: Standard deviation (default 1)
%
% Returns:
%   samples: Array of normal random samples
    if nargin < 2, mu = 0; end
    if nargin < 3, sigma = 1; end
    samples = mu + sigma * randn(n, 1);
end
function [samples, acceptance_rate] = metropolisHastings(target_log_pdf, ...
    initial_state, n_samples, proposal_cov)
% Metropolis-Hastings MCMC sampling
%
% Args:
%   target_log_pdf: Function handle for log target PDF
%   initial_state: Initial state vector
%   n_samples: Number of samples
%   proposal_cov: Proposal covariance matrix
%
% Returns:
%   samples: MCMC samples
%   acceptance_rate: Acceptance rate
    d = length(initial_state);
    samples = zeros(n_samples, d);
    current_state = initial_state(:);
    current_log_prob = target_log_pdf(current_state);
    n_accepted = 0;
    for i = 1:n_samples
        % Propose new state
        proposal = current_state + mvnrnd(zeros(d,1), proposal_cov)';
        proposal_log_prob = target_log_pdf(proposal);
        % Acceptance probability
        log_alpha = min(0, proposal_log_prob - current_log_prob);
        % Accept or reject
        if log(rand) < log_alpha
            current_state = proposal;
            current_log_prob = proposal_log_prob;
            n_accepted = n_accepted + 1;
        end
        samples(i, :) = current_state';
    end
    acceptance_rate = n_accepted / n_samples;
end
function [estimate, variance] = importanceSampling(target_pdf, ...
    proposal_pdf, proposal_sampler, integrand, n_samples)
% Importance sampling estimation
%
% Args:
%   target_pdf: Target PDF function handle
%   proposal_pdf: Proposal PDF function handle
%   proposal_sampler: Function to sample from proposal
%   integrand: Function to integrate
%   n_samples: Number of samples
%
% Returns:
%   estimate: Monte Carlo estimate
%   variance: Estimate variance
    % Generate samples from proposal
    samples = proposal_sampler(n_samples);
    % Calculate importance weights
    weights = target_pdf(samples) ./ proposal_pdf(samples);
    % Evaluate integrand
    values = integrand(samples);
    % Weighted average
    estimate = sum(weights .* values) / sum(weights);
    % Variance estimate
    variance = var(weights .* values) / n_samples;
end
function [samples, log_probs] = mcmcSampling(log_posterior, initial_state, ...
    n_samples, varargin)
% General MCMC sampling with adaptive proposal
%
% Args:
%   log_posterior: Log posterior function handle
%   initial_state: Initial parameter values
%   n_samples: Number of samples
%   varargin: Optional parameters
%
% Returns:
%   samples: MCMC samples
%   log_probs: Log probability values
    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'adapt_interval', 100, @isnumeric);
    addParameter(p, 'target_acceptance', 0.44, @isnumeric);
    addParameter(p, 'initial_step_size', 1.0, @isnumeric);
    parse(p, varargin{:});
    adapt_interval = p.Results.adapt_interval;
    target_acceptance = p.Results.target_acceptance;
    step_size = p.Results.initial_step_size;
    d = length(initial_state);
    samples = zeros(n_samples, d);
    log_probs = zeros(n_samples, 1);
    current_state = initial_state(:);
    current_log_prob = log_posterior(current_state);
    n_accepted = 0;
    for i = 1:n_samples
        % Adaptive step size
        if mod(i, adapt_interval) == 0 && i > adapt_interval
            acceptance_rate = n_accepted / adapt_interval;
            if acceptance_rate < target_acceptance
                step_size = step_size * 0.9;
            else
                step_size = step_size * 1.1;
            end
            n_accepted = 0;
        end
        % Propose new state
        proposal = current_state + step_size * randn(d, 1);
        proposal_log_prob = log_posterior(proposal);
        % Accept or reject
        if log(rand) < (proposal_log_prob - current_log_prob)
            current_state = proposal;
            current_log_prob = proposal_log_prob;
            n_accepted = n_accepted + 1;
        end
        samples(i, :) = current_state';
        log_probs(i) = current_log_prob;
    end
end
function samples = gibbsSampling(conditional_samplers, initial_state, n_samples)
% Gibbs sampling for multivariate distributions
%
% Args:
%   conditional_samplers: Cell array of conditional sampling functions
%   initial_state: Initial state vector
%   n_samples: Number of samples
%
% Returns:
%   samples: Gibbs samples
    d = length(initial_state);
    samples = zeros(n_samples, d);
    current_state = initial_state(:);
    for i = 1:n_samples
        % Sample each component conditionally
        for j = 1:d
            current_state(j) = conditional_samplers{j}(current_state);
        end
        samples(i, :) = current_state';
    end
end
function [integral_estimate, error_estimate] = monteCarloIntegration(...
    integrand, domain, n_samples, varargin)
% Monte Carlo integration
%
% Args:
%   integrand: Function to integrate
%   domain: Integration domain [a1 b1; a2 b2; ...]
%   n_samples: Number of samples
%   varargin: Optional parameters
%
% Returns:
%   integral_estimate: Estimated integral value
%   error_estimate: Error estimate
    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'method', 'uniform', @ischar);
    addParameter(p, 'importance_pdf', [], @(x) isa(x, 'function_handle'));
    parse(p, varargin{:});
    method = p.Results.method;
    importance_pdf = p.Results.importance_pdf;
    d = size(domain, 1);
    volume = prod(domain(:, 2) - domain(:, 1));
    if strcmp(method, 'uniform')
        % Uniform sampling
        samples = zeros(n_samples, d);
        for i = 1:d
            samples(:, i) = domain(i, 1) + ...
                (domain(i, 2) - domain(i, 1)) * rand(n_samples, 1);
        end
        % Evaluate integrand
        values = zeros(n_samples, 1);
        for i = 1:n_samples
            values(i) = integrand(samples(i, :));
        end
        integral_estimate = volume * mean(values);
        error_estimate = volume * std(values) / sqrt(n_samples);
    elseif strcmp(method, 'importance') && ~isempty(importance_pdf)
        % Importance sampling (implementation depends on specific case)
        error('Importance sampling integration not fully implemented');
    else
        error('Unknown integration method or missing importance PDF');
    end
end