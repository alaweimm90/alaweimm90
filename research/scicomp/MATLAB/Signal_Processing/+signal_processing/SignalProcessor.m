classdef SignalProcessor < handle
    % SignalProcessor - Core signal processing operations
    %
    % BERKELEY SCICOMP - SIGNAL PROCESSING TOOLBOX
    % =============================================
    %
    % Comprehensive signal processing toolkit including filtering,
    % spectral analysis, and feature extraction.
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties
        SamplingRate    % Sampling frequency (Hz)
        NyquistFreq     % Nyquist frequency
        % Berkeley colors
        BerkeleyBlue = [0 50 98]/255;
        CaliforniaGold = [253 181 21]/255;
    end
    methods
        function obj = SignalProcessor(sampling_rate)
            % Constructor for SignalProcessor
            %
            % Args:
            %   sampling_rate: Sampling frequency in Hz
            if nargin < 1
                sampling_rate = 1000;  % Default 1 kHz
            end
            obj.SamplingRate = sampling_rate;
            obj.NyquistFreq = sampling_rate / 2;
        end
        function [t, signal_data] = generateSignal(obj, signal_type, duration, varargin)
            % Generate various types of test signals
            %
            % Args:
            %   signal_type: 'sine', 'chirp', 'square', 'noise', 'multi_sine'
            %   duration: Signal duration in seconds
            %   varargin: Additional parameters (frequency, amplitude, etc.)
            % Parse optional inputs
            p = inputParser;
            addParameter(p, 'frequency', 10, @isnumeric);
            addParameter(p, 'amplitude', 1, @isnumeric);
            addParameter(p, 'phase', 0, @isnumeric);
            addParameter(p, 'noise_level', 0, @isnumeric);
            parse(p, varargin{:});
            % Generate time vector
            t = 0:1/obj.SamplingRate:duration-1/obj.SamplingRate;
            t = t(:);  % Column vector
            switch lower(signal_type)
                case 'sine'
                    signal_data = p.Results.amplitude * ...
                        sin(2*pi*p.Results.frequency*t + p.Results.phase);
                case 'chirp'
                    f0 = p.Results.frequency(1);
                    if length(p.Results.frequency) > 1
                        f1 = p.Results.frequency(2);
                    else
                        f1 = f0 * 10;  % Default to 10x start frequency
                    end
                    signal_data = p.Results.amplitude * ...
                        chirp(t, f0, duration, f1);
                case 'square'
                    signal_data = p.Results.amplitude * ...
                        square(2*pi*p.Results.frequency*t + p.Results.phase);
                case 'noise'
                    signal_data = p.Results.amplitude * randn(size(t));
                case 'multi_sine'
                    signal_data = zeros(size(t));
                    freqs = p.Results.frequency;
                    if isscalar(p.Results.amplitude)
                        amps = p.Results.amplitude * ones(size(freqs));
                    else
                        amps = p.Results.amplitude;
                    end
                    for i = 1:length(freqs)
                        signal_data = signal_data + ...
                            amps(i) * sin(2*pi*freqs(i)*t);
                    end
                otherwise
                    error('Unknown signal type: %s', signal_type);
            end
            % Add noise if specified
            if p.Results.noise_level > 0
                noise = p.Results.noise_level * randn(size(signal_data));
                signal_data = signal_data + noise;
            end
        end
        function [freq, mag] = computeFFT(obj, signal_data, varargin)
            % Compute Fast Fourier Transform
            %
            % Args:
            %   signal_data: Input signal
            %   varargin: Optional parameters (window, normalize)
            %
            % Returns:
            %   freq: Frequency vector
            %   mag: FFT magnitude
            p = inputParser;
            addParameter(p, 'window', 'none', @ischar);
            addParameter(p, 'normalize', true, @islogical);
            parse(p, varargin{:});
            N = length(signal_data);
            % Apply window if specified
            if ~strcmpi(p.Results.window, 'none')
                switch lower(p.Results.window)
                    case 'hann'
                        window = hann(N);
                    case 'hamming'
                        window = hamming(N);
                    case 'blackman'
                        window = blackman(N);
                    case 'kaiser'
                        window = kaiser(N);
                    otherwise
                        window = ones(N, 1);
                end
                signal_windowed = signal_data(:) .* window(:);
            else
                signal_windowed = signal_data(:);
            end
            % Compute FFT
            Y = fft(signal_windowed);
            % Single-sided spectrum
            if mod(N, 2) == 0
                Y = Y(1:N/2+1);
                freq = obj.SamplingRate * (0:N/2) / N;
            else
                Y = Y(1:(N+1)/2);
                freq = obj.SamplingRate * (0:(N-1)/2) / N;
            end
            % Magnitude
            mag = abs(Y);
            % Normalize if requested
            if p.Results.normalize
                mag = 2 * mag / N;
                mag(1) = mag(1) / 2;  % DC component
                if mod(N, 2) == 0
                    mag(end) = mag(end) / 2;  % Nyquist component
                end
            end
            freq = freq(:);
            mag = mag(:);
        end
        function [t, f, S] = computeSpectrogram(obj, signal_data, varargin)
            % Compute spectrogram using STFT
            %
            % Args:
            %   signal_data: Input signal
            %   varargin: window_size, overlap, window type
            %
            % Returns:
            %   t: Time vector
            %   f: Frequency vector
            %   S: Spectrogram magnitude
            p = inputParser;
            addParameter(p, 'window_size', 256, @isnumeric);
            addParameter(p, 'overlap', 0.5, @isnumeric);
            addParameter(p, 'window', 'hann', @ischar);
            parse(p, varargin{:});
            window_size = p.Results.window_size;
            overlap_samples = round(window_size * p.Results.overlap);
            % Use MATLAB's spectrogram function
            [S, f, t] = spectrogram(signal_data, window_size, ...
                overlap_samples, window_size, obj.SamplingRate);
            S = abs(S);  % Magnitude
        end
        function [b, a] = designFilter(obj, filter_type, cutoff, varargin)
            % Design digital filters
            %
            % Args:
            %   filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
            %   cutoff: Cutoff frequency/frequencies
            %   varargin: order, method
            %
            % Returns:
            %   b, a: Filter coefficients
            p = inputParser;
            addParameter(p, 'order', 4, @isnumeric);
            addParameter(p, 'method', 'butter', @ischar);
            parse(p, varargin{:});
            % Normalize frequencies to Nyquist
            Wn = cutoff / obj.NyquistFreq;
            switch lower(p.Results.method)
                case 'butter'
                    [b, a] = butter(p.Results.order, Wn, filter_type);
                case 'cheby1'
                    Rp = 0.5;  % Passband ripple (dB)
                    [b, a] = cheby1(p.Results.order, Rp, Wn, filter_type);
                case 'cheby2'
                    Rs = 40;  % Stopband attenuation (dB)
                    [b, a] = cheby2(p.Results.order, Rs, Wn, filter_type);
                case 'ellip'
                    Rp = 0.5;  % Passband ripple
                    Rs = 40;   % Stopband attenuation
                    [b, a] = ellip(p.Results.order, Rp, Rs, Wn, filter_type);
                otherwise
                    error('Unknown filter method: %s', p.Results.method);
            end
        end
        function filtered_signal = applyFilter(obj, signal_data, b, a, varargin)
            % Apply digital filter to signal
            %
            % Args:
            %   signal_data: Input signal
            %   b, a: Filter coefficients
            %   varargin: method ('filter' or 'filtfilt')
            p = inputParser;
            addParameter(p, 'method', 'filtfilt', @ischar);
            parse(p, varargin{:});
            switch lower(p.Results.method)
                case 'filter'
                    filtered_signal = filter(b, a, signal_data);
                case 'filtfilt'
                    filtered_signal = filtfilt(b, a, signal_data);
                otherwise
                    error('Unknown filtering method: %s', p.Results.method);
            end
        end
        function peaks = detectPeaks(obj, signal_data, varargin)
            % Detect peaks in signal
            %
            % Args:
            %   signal_data: Input signal
            %   varargin: Optional parameters
            %
            % Returns:
            %   peaks: Structure with peak information
            p = inputParser;
            addParameter(p, 'min_height', [], @isnumeric);
            addParameter(p, 'min_distance', 1, @isnumeric);
            addParameter(p, 'min_prominence', [], @isnumeric);
            parse(p, varargin{:});
            % Find peaks using findpeaks
            if ~isempty(p.Results.min_height) && ~isempty(p.Results.min_prominence)
                [pks, locs, widths, proms] = findpeaks(signal_data, ...
                    'MinPeakHeight', p.Results.min_height, ...
                    'MinPeakDistance', p.Results.min_distance, ...
                    'MinPeakProminence', p.Results.min_prominence);
            elseif ~isempty(p.Results.min_height)
                [pks, locs, widths, proms] = findpeaks(signal_data, ...
                    'MinPeakHeight', p.Results.min_height, ...
                    'MinPeakDistance', p.Results.min_distance);
            else
                [pks, locs, widths, proms] = findpeaks(signal_data, ...
                    'MinPeakDistance', p.Results.min_distance);
            end
            peaks.indices = locs;
            peaks.values = pks;
            peaks.widths = widths;
            peaks.prominences = proms;
        end
        function envelope = computeEnvelope(obj, signal_data, method)
            % Compute signal envelope
            %
            % Args:
            %   signal_data: Input signal
            %   method: 'hilbert' or 'peak'
            if nargin < 3
                method = 'hilbert';
            end
            switch lower(method)
                case 'hilbert'
                    % Hilbert transform method
                    analytic = hilbert(signal_data);
                    envelope = abs(analytic);
                case 'peak'
                    % Peak interpolation method
                    [upper, ~] = envelope(signal_data);
                    envelope = upper;
                otherwise
                    error('Unknown envelope method: %s', method);
            end
        end
        function [f, Pxx] = computePSD(obj, signal_data, varargin)
            % Compute Power Spectral Density
            %
            % Args:
            %   signal_data: Input signal
            %   varargin: method, window, etc.
            p = inputParser;
            addParameter(p, 'method', 'welch', @ischar);
            addParameter(p, 'window', [], @isnumeric);
            addParameter(p, 'overlap', [], @isnumeric);
            parse(p, varargin{:});
            switch lower(p.Results.method)
                case 'periodogram'
                    [Pxx, f] = periodogram(signal_data, [], [], obj.SamplingRate);
                case 'welch'
                    if isempty(p.Results.window)
                        window = min(256, length(signal_data));
                    else
                        window = p.Results.window;
                    end
                    if isempty(p.Results.overlap)
                        overlap = round(window/2);
                    else
                        overlap = p.Results.overlap;
                    end
                    [Pxx, f] = pwelch(signal_data, window, overlap, ...
                        window, obj.SamplingRate);
                otherwise
                    error('Unknown PSD method: %s', p.Results.method);
            end
        end
        function correlation = computeCorrelation(obj, signal1, signal2, varargin)
            % Compute cross-correlation
            %
            % Args:
            %   signal1, signal2: Input signals
            %   varargin: normalize flag
            p = inputParser;
            addParameter(p, 'normalize', true, @islogical);
            parse(p, varargin{:});
            % Compute cross-correlation
            correlation = xcorr(signal1, signal2);
            % Normalize if requested
            if p.Results.normalize
                norm_factor = sqrt(sum(signal1.^2) * sum(signal2.^2));
                if norm_factor > 0
                    correlation = correlation / norm_factor;
                end
            end
        end
        function features = extractFeatures(obj, signal_data)
            % Extract time and frequency domain features
            %
            % Args:
            %   signal_data: Input signal
            %
            % Returns:
            %   features: Structure with extracted features
            % Time-domain features
            features.mean = mean(signal_data);
            features.std = std(signal_data);
            features.variance = var(signal_data);
            features.rms = rms(signal_data);
            features.peak_to_peak = peak2peak(signal_data);
            features.crest_factor = max(abs(signal_data)) / rms(signal_data);
            features.skewness = skewness(signal_data);
            features.kurtosis = kurtosis(signal_data);
            % Zero crossings
            zc = diff(sign(signal_data));
            features.zero_crossings = sum(abs(zc) > 0) / 2;
            % Frequency-domain features
            [freq, mag] = obj.computeFFT(signal_data);
            % Spectral centroid
            if sum(mag) > 0
                features.spectral_centroid = sum(freq .* mag) / sum(mag);
            else
                features.spectral_centroid = 0;
            end
            % Dominant frequency
            [~, idx] = max(mag);
            features.dominant_frequency = freq(idx);
            features.dominant_magnitude = mag(idx);
            % Spectral bandwidth
            if sum(mag) > 0
                mean_freq = features.spectral_centroid;
                features.spectral_bandwidth = sqrt(...
                    sum(((freq - mean_freq).^2) .* mag) / sum(mag));
            else
                features.spectral_bandwidth = 0;
            end
        end
        function snr_db = computeSNR(obj, signal_clean, signal_noisy)
            % Compute Signal-to-Noise Ratio
            %
            % Args:
            %   signal_clean: Clean signal
            %   signal_noisy: Noisy signal
            %
            % Returns:
            %   snr_db: SNR in decibels
            % Extract noise
            noise = signal_noisy - signal_clean;
            % Compute powers
            signal_power = mean(signal_clean.^2);
            noise_power = mean(noise.^2);
            % SNR in dB
            if noise_power > 0
                snr_db = 10 * log10(signal_power / noise_power);
            else
                snr_db = Inf;
            end
        end
        function plotSignal(obj, t, signal_data, varargin)
            % Plot signal with Berkeley styling
            %
            % Args:
            %   t: Time vector
            %   signal_data: Signal to plot
            %   varargin: title, labels, etc.
            p = inputParser;
            addParameter(p, 'title', 'Signal', @ischar);
            addParameter(p, 'xlabel', 'Time (s)', @ischar);
            addParameter(p, 'ylabel', 'Amplitude', @ischar);
            parse(p, varargin{:});
            plot(t, signal_data, 'Color', obj.BerkeleyBlue, 'LineWidth', 1.5);
            xlabel(p.Results.xlabel);
            ylabel(p.Results.ylabel);
            title(p.Results.title, 'Color', obj.BerkeleyBlue);
            grid on;
            set(gca, 'GridAlpha', 0.3);
        end
        function plotSpectrum(obj, freq, mag, varargin)
            % Plot frequency spectrum with Berkeley styling
            %
            % Args:
            %   freq: Frequency vector
            %   mag: Magnitude spectrum
            %   varargin: plot options
            p = inputParser;
            addParameter(p, 'title', 'Frequency Spectrum', @ischar);
            addParameter(p, 'scale', 'linear', @ischar);
            parse(p, varargin{:});
            if strcmpi(p.Results.scale, 'log')
                semilogy(freq, mag, 'Color', obj.CaliforniaGold, 'LineWidth', 1.5);
            else
                plot(freq, mag, 'Color', obj.CaliforniaGold, 'LineWidth', 1.5);
            end
            xlabel('Frequency (Hz)');
            ylabel('Magnitude');
            title(p.Results.title, 'Color', obj.BerkeleyBlue);
            grid on;
            set(gca, 'GridAlpha', 0.3);
            xlim([0 obj.NyquistFreq]);
        end
    end
end