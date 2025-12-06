function varargout = error_handling(operation, varargin)
% ERROR_HANDLING - Comprehensive error handling utilities for SpinCirc
%
% This function provides robust error handling, logging, and debugging
% utilities for spintronic device simulations with detailed error reporting
% and recovery mechanisms.
%
% Syntax:
%   result = error_handling(operation, ...)
%   [result1, result2, ...] = error_handling(operation, ...)
%
% Operations:
%   'try_catch'       - Enhanced try-catch with logging
%   'validate_input'  - Input parameter validation
%   'assert_range'    - Assert values within specified range
%   'check_dimensions' - Check matrix/array dimensions
%   'log_error'       - Log errors with detailed information
%   'log_warning'     - Log warnings with context
%   'create_logger'   - Create custom logger instance
%   'debug_info'      - Generate debug information
%   'stack_trace'     - Enhanced stack trace information
%   'recovery'        - Error recovery mechanisms
%   'validate_file'   - File existence and format validation
%   'memory_check'    - Memory usage monitoring
%
% Examples:
%   % Enhanced try-catch
%   result = error_handling('try_catch', @my_function, args, 'fallback', default_value)
%   
%   % Input validation
%   error_handling('validate_input', value, 'numeric', 'positive', 'finite')
%   
%   % Range assertion
%   error_handling('assert_range', value, [0, 1], 'parameter_name')
%   
%   % Create logger
%   logger = error_handling('create_logger', 'simulation.log')
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    if nargin < 1
        error('Operation must be specified');
    end
    
    switch lower(operation)
        case 'try_catch'
            [varargout{1:nargout}] = enhanced_try_catch(varargin{:});
            
        case 'validate_input'
            [varargout{1:nargout}] = validate_input(varargin{:});
            
        case 'assert_range'
            [varargout{1:nargout}] = assert_range(varargin{:});
            
        case 'check_dimensions'
            [varargout{1:nargout}] = check_dimensions(varargin{:});
            
        case 'log_error'
            [varargout{1:nargout}] = log_error(varargin{:});
            
        case 'log_warning'
            [varargout{1:nargout}] = log_warning(varargin{:});
            
        case 'create_logger'
            [varargout{1:nargout}] = create_logger(varargin{:});
            
        case 'debug_info'
            [varargout{1:nargout}] = generate_debug_info(varargin{:});
            
        case 'stack_trace'
            [varargout{1:nargout}] = enhanced_stack_trace(varargin{:});
            
        case 'recovery'
            [varargout{1:nargout}] = error_recovery(varargin{:});
            
        case 'validate_file'
            [varargout{1:nargout}] = validate_file(varargin{:});
            
        case 'memory_check'
            [varargout{1:nargout}] = memory_check(varargin{:});
            
        otherwise
            error('Unknown operation: %s', operation);
    end
end

function varargout = enhanced_try_catch(func_handle, func_args, varargin)
    % Enhanced try-catch with detailed error handling
    
    p = inputParser;
    addRequired(p, 'func_handle', @(x) isa(x, 'function_handle'));
    addRequired(p, 'func_args', @iscell);
    addParameter(p, 'Fallback', [], @(x) true);
    addParameter(p, 'LogErrors', true, @islogical);
    addParameter(p, 'Retries', 0, @(x) isscalar(x) && x >= 0);
    addParameter(p, 'RetryDelay', 1, @(x) isscalar(x) && x > 0);
    addParameter(p, 'ErrorHandler', [], @(x) isa(x, 'function_handle'));
    addParameter(p, 'Context', '', @(x) ischar(x) || isstring(x));
    
    parse(p, func_handle, func_args, varargin{:});
    
    fallback = p.Results.Fallback;
    log_errors = p.Results.LogErrors;
    retries = p.Results.Retries;
    retry_delay = p.Results.RetryDelay;
    error_handler = p.Results.ErrorHandler;
    context = p.Results.Context;
    
    last_error = [];
    attempt = 0;
    max_attempts = retries + 1;
    
    while attempt < max_attempts
        attempt = attempt + 1;
        
        try
            % Execute function
            if isempty(func_args)
                [varargout{1:nargout}] = func_handle();
            else
                [varargout{1:nargout}] = func_handle(func_args{:});
            end
            
            return; % Success - exit function
            
        catch ME
            last_error = ME;
            
            % Log error details
            if log_errors
                error_info = struct();
                error_info.message = ME.message;
                error_info.identifier = ME.identifier;
                error_info.stack = ME.stack;
                error_info.attempt = attempt;
                error_info.context = context;
                error_info.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
                
                log_error('Function execution failed', error_info);
            end
            
            % Custom error handler
            if ~isempty(error_handler)
                try
                    recovery_action = error_handler(ME, attempt, context);
                    if strcmpi(recovery_action, 'continue')
                        continue;
                    elseif strcmpi(recovery_action, 'abort')
                        break;
                    end
                catch
                    % Error in error handler - continue with normal flow
                end
            end
            
            % Retry delay
            if attempt < max_attempts
                pause(retry_delay);
            end
        end
    end
    
    % All attempts failed
    if ~isempty(fallback)
        if iscell(fallback)
            [varargout{1:nargout}] = fallback{:};
        else
            varargout{1} = fallback;
        end
        
        if log_errors
            log_warning(sprintf('Function failed after %d attempts, using fallback', max_attempts), ...
                       struct('context', context, 'final_error', last_error));
        end
    else
        % Re-throw the last error with enhanced information
        enhanced_error = MException('SpinCirc:EnhancedTryCatch', ...
                                   'Function failed after %d attempts in context: %s', ...
                                   max_attempts, context);
        enhanced_error = enhanced_error.addCause(last_error);
        throw(enhanced_error);
    end
end

function validate_input(value, varargin)
    % Comprehensive input validation
    
    if nargin < 2
        error('At least one validation criterion must be specified');
    end
    
    criteria = varargin;
    
    for i = 1:length(criteria)
        criterion = criteria{i};
        
        switch lower(criterion)
            case 'numeric'
                if ~isnumeric(value)
                    error('SpinCirc:InvalidInput', 'Value must be numeric');
                end
                
            case 'scalar'
                if ~isscalar(value)
                    error('SpinCirc:InvalidInput', 'Value must be scalar');
                end
                
            case 'vector'
                if ~isvector(value)
                    error('SpinCirc:InvalidInput', 'Value must be a vector');
                end
                
            case 'matrix'
                if ~ismatrix(value) || isvector(value)
                    error('SpinCirc:InvalidInput', 'Value must be a matrix');
                end
                
            case 'positive'
                if ~all(value(:) > 0)
                    error('SpinCirc:InvalidInput', 'All values must be positive');
                end
                
            case 'nonnegative'
                if ~all(value(:) >= 0)
                    error('SpinCirc:InvalidInput', 'All values must be non-negative');
                end
                
            case 'finite'
                if ~all(isfinite(value(:)))
                    error('SpinCirc:InvalidInput', 'All values must be finite');
                end
                
            case 'real'
                if ~isreal(value)
                    error('SpinCirc:InvalidInput', 'Value must be real');
                end
                
            case 'integer'
                if ~all(value(:) == round(value(:)))
                    error('SpinCirc:InvalidInput', 'All values must be integers');
                end
                
            case 'nonempty'
                if isempty(value)
                    error('SpinCirc:InvalidInput', 'Value cannot be empty');
                end
                
            case 'square'
                if ~ismatrix(value) || size(value, 1) ~= size(value, 2)
                    error('SpinCirc:InvalidInput', 'Matrix must be square');
                end
                
            case 'symmetric'
                if ~ismatrix(value) || ~issymmetric(value)
                    error('SpinCirc:InvalidInput', 'Matrix must be symmetric');
                end
                
            case 'hermitian'
                if ~ismatrix(value) || ~ishermitian(value)
                    error('SpinCirc:InvalidInput', 'Matrix must be Hermitian');
                end
                
            otherwise
                warning('Unknown validation criterion: %s', criterion);
        end
    end
end

function assert_range(value, range, param_name, varargin)
    % Assert values are within specified range
    
    p = inputParser;
    addRequired(p, 'value', @isnumeric);
    addRequired(p, 'range', @(x) isnumeric(x) && length(x) == 2);
    addRequired(p, 'param_name', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Inclusive', true, @islogical);
    addParameter(p, 'AllowNaN', false, @islogical);
    
    parse(p, value, range, param_name, varargin{:});
    
    inclusive = p.Results.Inclusive;
    allow_nan = p.Results.AllowNaN;
    
    % Handle NaN values
    if ~allow_nan && any(isnan(value(:)))
        error('SpinCirc:OutOfRange', 'Parameter %s contains NaN values', param_name);
    end
    
    % Remove NaN values for range checking
    valid_values = value(~isnan(value));
    
    if isempty(valid_values)
        return; % All NaN, already handled above
    end
    
    min_val = min(range);
    max_val = max(range);
    
    if inclusive
        condition = (valid_values >= min_val) & (valid_values <= max_val);
    else
        condition = (valid_values > min_val) & (valid_values < max_val);
    end
    
    if ~all(condition)
        bound_type = iif(inclusive, 'inclusive', 'exclusive');
        error('SpinCirc:OutOfRange', ...
              'Parameter %s must be within range [%.3g, %.3g] (%s)', ...
              param_name, min_val, max_val, bound_type);
    end
end

function check_dimensions(value, expected_dims, param_name, varargin)
    % Check array dimensions
    
    p = inputParser;
    addRequired(p, 'value');
    addRequired(p, 'expected_dims', @isnumeric);
    addRequired(p, 'param_name', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Exact', true, @islogical);
    addParameter(p, 'AllowSqueeze', false, @islogical);
    
    parse(p, value, expected_dims, param_name, varargin{:});
    
    exact = p.Results.Exact;
    allow_squeeze = p.Results.AllowSqueeze;
    
    actual_dims = size(value);
    
    if allow_squeeze
        actual_dims = actual_dims(actual_dims > 1);
        expected_dims = expected_dims(expected_dims > 1);
    end
    
    if exact
        if ~isequal(actual_dims, expected_dims)
            error('SpinCirc:DimensionMismatch', ...
                  'Parameter %s has dimensions [%s], expected [%s]', ...
                  param_name, num2str(actual_dims), num2str(expected_dims));
        end
    else
        if length(actual_dims) ~= length(expected_dims)
            error('SpinCirc:DimensionMismatch', ...
                  'Parameter %s has %d dimensions, expected %d', ...
                  param_name, length(actual_dims), length(expected_dims));
        end
    end
end

function log_error(message, error_info, varargin)
    % Log error with detailed information
    
    p = inputParser;
    addRequired(p, 'message', @(x) ischar(x) || isstring(x));
    addRequired(p, 'error_info');
    addParameter(p, 'LogFile', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Display', true, @islogical);
    addParameter(p, 'Level', 'ERROR', @(x) ischar(x) || isstring(x));
    
    parse(p, message, error_info, varargin{:});
    
    log_file = p.Results.LogFile;
    display_log = p.Results.Display;
    level = p.Results.Level;
    
    % Create log entry
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
    log_entry = sprintf('[%s] %s: %s', timestamp, level, message);
    
    % Add error details
    if isstruct(error_info)
        fields = fieldnames(error_info);
        for i = 1:length(fields)
            field_name = fields{i};
            field_value = error_info.(field_name);
            
            if ischar(field_value) || isstring(field_value)
                log_entry = sprintf('%s\n  %s: %s', log_entry, field_name, field_value);
            elseif isnumeric(field_value) && isscalar(field_value)
                log_entry = sprintf('%s\n  %s: %.6g', log_entry, field_name, field_value);
            elseif isstruct(field_value) && strcmp(field_name, 'stack')
                % Special handling for stack trace
                log_entry = sprintf('%s\n  Stack Trace:', log_entry);
                for j = 1:length(field_value)
                    log_entry = sprintf('%s\n    %s (line %d)', log_entry, ...
                                      field_value(j).name, field_value(j).line);
                end
            end
        end
    elseif ischar(error_info) || isstring(error_info)
        log_entry = sprintf('%s\n  Details: %s', log_entry, error_info);
    end
    
    % Display to command window
    if display_log
        fprintf(2, '%s\n', log_entry); % stderr
    end
    
    % Write to log file
    if ~isempty(log_file)
        fid = fopen(log_file, 'a');
        if fid > 0
            fprintf(fid, '%s\n\n', log_entry);
            fclose(fid);
        end
    end
end

function log_warning(message, context, varargin)
    % Log warning with context
    
    warning_info = struct();
    warning_info.context = context;
    warning_info.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    log_error(message, warning_info, 'Level', 'WARNING', varargin{:});
end

function logger = create_logger(log_file, varargin)
    % Create custom logger instance
    
    p = inputParser;
    addRequired(p, 'log_file', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Level', 'INFO', @(x) ischar(x) || isstring(x));
    addParameter(p, 'MaxFileSize', 10e6, @(x) isscalar(x) && x > 0); % 10 MB
    addParameter(p, 'RotateFiles', true, @islogical);
    
    parse(p, log_file, varargin{:});
    
    level = p.Results.Level;
    max_file_size = p.Results.MaxFileSize;
    rotate_files = p.Results.RotateFiles;
    
    logger = struct();
    logger.file = log_file;
    logger.level = level;
    logger.max_size = max_file_size;
    logger.rotate = rotate_files;
    logger.created = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    % Create log directory if it doesn't exist
    [log_dir, ~, ~] = fileparts(log_file);
    if ~isempty(log_dir) && ~exist(log_dir, 'dir')
        mkdir(log_dir);
    end
    
    % Initialize log file
    fid = fopen(log_file, 'a');
    if fid > 0
        fprintf(fid, '\n=== SpinCirc Log Session Started: %s ===\n', logger.created);
        fclose(fid);
    end
end

function debug_info = generate_debug_info(varargin)
    % Generate comprehensive debug information
    
    debug_info = struct();
    
    % System information
    debug_info.system.matlab_version = version('-release');
    debug_info.system.platform = computer;
    debug_info.system.java_version = char(java.lang.System.getProperty('java.version'));
    debug_info.system.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    % Memory information
    if ispc
        [~, sys_info] = memory;
        debug_info.memory.physical_total = sys_info.PhysicalMemory.Total;
        debug_info.memory.physical_available = sys_info.PhysicalMemory.Available;
        debug_info.memory.virtual_total = sys_info.VirtualAddressSpace.Total;
        debug_info.memory.virtual_available = sys_info.VirtualAddressSpace.Available;
    end
    
    % MATLAB workspace information
    workspace_vars = evalin('base', 'whos');
    debug_info.workspace.num_variables = length(workspace_vars);
    if ~isempty(workspace_vars)
        debug_info.workspace.total_size = sum([workspace_vars.bytes]);
        debug_info.workspace.largest_var = workspace_vars(1).name;
        [~, max_idx] = max([workspace_vars.bytes]);
        debug_info.workspace.largest_var = workspace_vars(max_idx).name;
    end
    
    % Current function stack
    stack = dbstack;
    debug_info.call_stack = stack;
    
    % Path information
    debug_info.path.current_dir = pwd;
    debug_info.path.matlab_path = path;
    
    % SpinCirc specific information
    try
        debug_info.spincirc.version = get_spincirc_version();
    catch
        debug_info.spincirc.version = 'unknown';
    end
end

function stack_info = enhanced_stack_trace(ME, varargin)
    % Enhanced stack trace with additional context
    
    p = inputParser;
    addRequired(p, 'ME', @(x) isa(x, 'MException'));
    addParameter(p, 'Context', true, @islogical);
    addParameter(p, 'Variables', false, @islogical);
    
    parse(p, ME, varargin{:});
    
    show_context = p.Results.Context;
    show_variables = p.Results.Variables;
    
    stack_info = struct();
    stack_info.error = ME;
    stack_info.enhanced_stack = [];
    
    if ~isempty(ME.stack)
        for i = 1:length(ME.stack)
            frame = ME.stack(i);
            enhanced_frame = frame;
            
            if show_context
                try
                    % Read source file around error line
                    if exist(frame.file, 'file')
                        fid = fopen(frame.file, 'r');
                        lines = {};
                        while ~feof(fid)
                            lines{end+1} = fgetl(fid);
                        end
                        fclose(fid);
                        
                        % Extract context around error line
                        start_line = max(1, frame.line - 3);
                        end_line = min(length(lines), frame.line + 3);
                        
                        enhanced_frame.context = lines(start_line:end_line);
                        enhanced_frame.context_start = start_line;
                    end
                catch
                    % Ignore context extraction errors
                end
            end
            
            stack_info.enhanced_stack = [stack_info.enhanced_stack; enhanced_frame];
        end
    end
end

function recovery_result = error_recovery(strategy, error_info, varargin)
    % Error recovery mechanisms
    
    p = inputParser;
    addRequired(p, 'strategy', @(x) ischar(x) || isstring(x));
    addRequired(p, 'error_info');
    addParameter(p, 'FallbackValue', [], @(x) true);
    addParameter(p, 'RetryCount', 3, @(x) isscalar(x) && x >= 0);
    
    parse(p, strategy, error_info, varargin{:});
    
    fallback_value = p.Results.FallbackValue;
    retry_count = p.Results.RetryCount;
    
    recovery_result = struct();
    recovery_result.success = false;
    recovery_result.action = 'none';
    recovery_result.value = [];
    
    switch lower(strategy)
        case 'fallback'
            recovery_result.success = true;
            recovery_result.action = 'fallback';
            recovery_result.value = fallback_value;
            
        case 'retry'
            recovery_result.success = true;
            recovery_result.action = 'retry';
            recovery_result.value = retry_count;
            
        case 'skip'
            recovery_result.success = true;
            recovery_result.action = 'skip';
            
        case 'abort'
            recovery_result.success = false;
            recovery_result.action = 'abort';
            
        otherwise
            warning('Unknown recovery strategy: %s', strategy);
    end
end

function valid = validate_file(filename, varargin)
    % Validate file existence and properties
    
    p = inputParser;
    addRequired(p, 'filename', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Extension', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'MinSize', 0, @(x) isscalar(x) && x >= 0);
    addParameter(p, 'MaxSize', Inf, @(x) isscalar(x) && x > 0);
    addParameter(p, 'Readable', true, @islogical);
    addParameter(p, 'Writable', false, @islogical);
    
    parse(p, filename, varargin{:});
    
    extension = p.Results.Extension;
    min_size = p.Results.MinSize;
    max_size = p.Results.MaxSize;
    readable = p.Results.Readable;
    writable = p.Results.Writable;
    
    valid = false;
    
    % Check existence
    if ~exist(filename, 'file')
        error('SpinCirc:FileNotFound', 'File does not exist: %s', filename);
    end
    
    % Check extension
    if ~isempty(extension)
        [~, ~, file_ext] = fileparts(filename);
        if ~strcmpi(file_ext, extension)
            error('SpinCirc:InvalidFileType', 'Expected %s file, got %s', extension, file_ext);
        end
    end
    
    % Check file size
    file_info = dir(filename);
    if file_info.bytes < min_size
        error('SpinCirc:FileTooSmall', 'File is too small (%.0f bytes, minimum %.0f)', ...
              file_info.bytes, min_size);
    end
    
    if file_info.bytes > max_size
        error('SpinCirc:FileTooLarge', 'File is too large (%.0f bytes, maximum %.0f)', ...
              file_info.bytes, max_size);
    end
    
    % Check permissions (simplified)
    if readable
        fid = fopen(filename, 'r');
        if fid < 0
            error('SpinCirc:FileNotReadable', 'File is not readable: %s', filename);
        end
        fclose(fid);
    end
    
    if writable
        fid = fopen(filename, 'a');
        if fid < 0
            error('SpinCirc:FileNotWritable', 'File is not writable: %s', filename);
        end
        fclose(fid);
    end
    
    valid = true;
end

function mem_info = memory_check(varargin)
    % Monitor memory usage
    
    p = inputParser;
    addParameter(p, 'Threshold', 0.9, @(x) isscalar(x) && x > 0 && x <= 1);
    addParameter(p, 'Action', 'warn', @(x) any(strcmpi(x, {'warn', 'error', 'none'})));
    
    parse(p, varargin{:});
    
    threshold = p.Results.Threshold;
    action = p.Results.Action;
    
    mem_info = struct();
    
    if ispc
        [user_mem, sys_mem] = memory;
        mem_info.used = user_mem.MemUsedMATLAB;
        mem_info.available = user_mem.MemAvailableAllArrays;
        mem_info.total = sys_mem.PhysicalMemory.Total;
        mem_info.usage_ratio = mem_info.used / mem_info.total;
    else
        % Simplified for other platforms
        mem_info.used = NaN;
        mem_info.available = NaN;
        mem_info.total = NaN;
        mem_info.usage_ratio = NaN;
    end
    
    % Check threshold
    if ~isnan(mem_info.usage_ratio) && mem_info.usage_ratio > threshold
        message = sprintf('Memory usage is %.1f%% (threshold: %.1f%%)', ...
                         mem_info.usage_ratio * 100, threshold * 100);
        
        switch lower(action)
            case 'warn'
                warning('SpinCirc:HighMemoryUsage', message);
            case 'error'
                error('SpinCirc:HighMemoryUsage', message);
            case 'none'
                % Do nothing
        end
    end
end

% Helper functions
function version = get_spincirc_version()
    % Get SpinCirc version from CHANGELOG or version file
    try
        % Try to read version from CHANGELOG.md
        changelog_path = fullfile(fileparts(which('error_handling')), '..', '..', '..', 'CHANGELOG.md');
        if exist(changelog_path, 'file')
            fid = fopen(changelog_path, 'r');
            first_line = fgetl(fid);
            fclose(fid);
            
            % Extract version from changelog (assumes format like "## [1.0.0]")
            tokens = regexp(first_line, '\[([^\]]+)\]', 'tokens');
            if ~isempty(tokens)
                version = tokens{1}{1};
                return;
            end
        end
        
        % Fallback version
        version = '1.0.0';
        
    catch
        version = '1.0.0';
    end
end

function result = iif(condition, true_value, false_value)
    % Inline if function
    if condition
        result = true_value;
    else
        result = false_value;
    end
end