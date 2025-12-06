function varargout = data_io(operation, varargin)
% DATA_IO - Comprehensive data input/output utilities for SpinCirc
%
% This function provides robust data I/O operations for spintronic device
% simulations including file reading/writing, data validation, format
% conversion, and metadata management.
%
% Syntax:
%   result = data_io(operation, ...)
%   [result1, result2, ...] = data_io(operation, ...)
%
% Operations:
%   'save'           - Save simulation data with metadata
%   'load'           - Load simulation data with validation
%   'export'         - Export data to various formats
%   'import'         - Import data from external formats
%   'validate'       - Validate data structure and values
%   'convert'        - Convert between data formats
%   'merge'          - Merge multiple datasets
%   'extract'        - Extract subset of data
%   'compress'       - Compress large datasets
%   'backup'         - Create backup copies
%   'metadata'       - Manage metadata operations
%   'batch_process'  - Process multiple files
%
% Supported Formats:
%   MAT-files, CSV, JSON, HDF5, Excel, ASCII, Binary
%
% Examples:
%   % Save simulation results
%   data_io('save', results, 'simulation_001.mat')
%   
%   % Load with validation
%   data = data_io('load', 'simulation_001.mat', 'validate', true)
%   
%   % Export to CSV
%   data_io('export', results, 'output.csv', 'format', 'csv')
%   
%   % Batch process multiple files
%   data_io('batch_process', '*.mat', @process_function)
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    if nargin < 1
        error('Operation must be specified');
    end
    
    switch lower(operation)
        case 'save'
            [varargout{1:nargout}] = save_data(varargin{:});
            
        case 'load'
            [varargout{1:nargout}] = load_data(varargin{:});
            
        case 'export'
            [varargout{1:nargout}] = export_data(varargin{:});
            
        case 'import'
            [varargout{1:nargout}] = import_data(varargin{:});
            
        case 'validate'
            [varargout{1:nargout}] = validate_data(varargin{:});
            
        case 'convert'
            [varargout{1:nargout}] = convert_data(varargin{:});
            
        case 'merge'
            [varargout{1:nargout}] = merge_data(varargin{:});
            
        case 'extract'
            [varargout{1:nargout}] = extract_data(varargin{:});
            
        case 'compress'
            [varargout{1:nargout}] = compress_data(varargin{:});
            
        case 'backup'
            [varargout{1:nargout}] = backup_data(varargin{:});
            
        case 'metadata'
            [varargout{1:nargout}] = manage_metadata(varargin{:});
            
        case 'batch_process'
            [varargout{1:nargout}] = batch_process(varargin{:});
            
        otherwise
            error('Unknown operation: %s', operation);
    end
end

function success = save_data(data, filename, varargin)
    % Save data with comprehensive metadata and validation
    
    p = inputParser;
    addRequired(p, 'data');
    addRequired(p, 'filename', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Metadata', struct(), @isstruct);
    addParameter(p, 'Compress', true, @islogical);
    addParameter(p, 'Validate', true, @islogical);
    addParameter(p, 'Backup', false, @islogical);
    addParameter(p, 'Format', 'auto', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Version', 'v7.3', @(x) ischar(x) || isstring(x));
    
    parse(p, data, filename, varargin{:});
    
    metadata = p.Results.Metadata;
    compress = p.Results.Compress;
    validate = p.Results.Validate;
    backup = p.Results.Backup;
    format = p.Results.Format;
    version = p.Results.Version;
    
    try
        % Data validation
        if validate
            validation_result = validate_data(data);
            if ~validation_result.valid
                warning('Data validation failed: %s', validation_result.message);
            end
        end
        
        % Create backup if requested
        if backup && exist(filename, 'file')
            backup_data(filename);
        end
        
        % Add automatic metadata
        metadata.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        metadata.matlab_version = version('-release');
        metadata.spincirc_version = get_spincirc_version();
        metadata.data_size = whos('data');
        metadata.checksum = calculate_checksum(data);
        
        % Determine format from extension
        [~, ~, ext] = fileparts(filename);
        if strcmpi(format, 'auto')
            format = determine_format(ext);
        end
        
        % Save based on format
        switch lower(format)
            case 'mat'
                if compress
                    save(filename, 'data', 'metadata', '-v7.3', '-nocompression');
                else
                    save(filename, 'data', 'metadata', version);
                end
                
            case 'json'
                save_json(filename, data, metadata);
                
            case 'hdf5'
                save_hdf5(filename, data, metadata);
                
            case 'csv'
                save_csv(filename, data, metadata);
                
            otherwise
                error('Unsupported format: %s', format);
        end
        
        success = true;
        
    catch ME
        warning('Failed to save data: %s', ME.message);
        success = false;
        rethrow(ME);
    end
end

function [data, metadata] = load_data(filename, varargin)
    % Load data with validation and format detection
    
    p = inputParser;
    addRequired(p, 'filename', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Validate', true, @islogical);
    addParameter(p, 'Format', 'auto', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Variables', {}, @iscell);
    addParameter(p, 'Partial', false, @islogical);
    
    parse(p, filename, varargin{:});
    
    validate = p.Results.Validate;
    format = p.Results.Format;
    variables = p.Results.Variables;
    partial = p.Results.Partial;
    
    if ~exist(filename, 'file')
        error('File does not exist: %s', filename);
    end
    
    % Determine format
    [~, ~, ext] = fileparts(filename);
    if strcmpi(format, 'auto')
        format = determine_format(ext);
    end
    
    try
        % Load based on format
        switch lower(format)
            case 'mat'
                if isempty(variables)
                    loaded = load(filename);
                else
                    loaded = load(filename, variables{:});
                end
                
                data = loaded.data;
                metadata = loaded.metadata;
                
            case 'json'
                [data, metadata] = load_json(filename);
                
            case 'hdf5'
                [data, metadata] = load_hdf5(filename);
                
            case 'csv'
                [data, metadata] = load_csv(filename);
                
            otherwise
                error('Unsupported format: %s', format);
        end
        
        % Validate loaded data
        if validate
            validation_result = validate_data(data);
            if ~validation_result.valid
                warning('Loaded data validation failed: %s', validation_result.message);
            end
        end
        
    catch ME
        error('Failed to load data from %s: %s', filename, ME.message);
    end
end

function success = export_data(data, filename, varargin)
    % Export data to various external formats
    
    p = inputParser;
    addRequired(p, 'data');
    addRequired(p, 'filename', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Format', 'csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Precision', 6, @(x) isscalar(x) && x > 0);
    addParameter(p, 'Headers', true, @islogical);
    addParameter(p, 'Delimiter', ',', @(x) ischar(x) || isstring(x));
    
    parse(p, data, filename, varargin{:});
    
    format = p.Results.Format;
    precision = p.Results.Precision;
    headers = p.Results.Headers;
    delimiter = p.Results.Delimiter;
    
    try
        switch lower(format)
            case 'csv'
                export_csv(data, filename, precision, headers, delimiter);
                
            case 'excel'
                export_excel(data, filename, headers);
                
            case 'ascii'
                export_ascii(data, filename, precision);
                
            case 'json'
                export_json(data, filename);
                
            otherwise
                error('Unsupported export format: %s', format);
        end
        
        success = true;
        
    catch ME
        warning('Failed to export data: %s', ME.message);
        success = false;
        rethrow(ME);
    end
end

function data = import_data(filename, varargin)
    % Import data from external formats
    
    p = inputParser;
    addRequired(p, 'filename', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Format', 'auto', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Delimiter', 'auto', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Headers', true, @islogical);
    addParameter(p, 'Sheet', 1, @(x) isscalar(x) && x > 0);
    
    parse(p, filename, varargin{:});
    
    format = p.Results.Format;
    delimiter = p.Results.Delimiter;
    headers = p.Results.Headers;
    sheet = p.Results.Sheet;
    
    if ~exist(filename, 'file')
        error('File does not exist: %s', filename);
    end
    
    % Auto-detect format
    [~, ~, ext] = fileparts(filename);
    if strcmpi(format, 'auto')
        format = determine_format(ext);
    end
    
    try
        switch lower(format)
            case 'csv'
                data = import_csv(filename, delimiter, headers);
                
            case 'excel'
                data = import_excel(filename, sheet, headers);
                
            case 'ascii'
                data = import_ascii(filename);
                
            case 'json'
                data = import_json(filename);
                
            otherwise
                error('Unsupported import format: %s', format);
        end
        
    catch ME
        error('Failed to import data: %s', ME.message);
    end
end

function result = validate_data(data, varargin)
    % Comprehensive data validation
    
    p = inputParser;
    addRequired(p, 'data');
    addParameter(p, 'Schema', struct(), @isstruct);
    addParameter(p, 'Strict', false, @islogical);
    addParameter(p, 'Numeric', true, @islogical);
    
    parse(p, data, varargin{:});
    
    schema = p.Results.Schema;
    strict = p.Results.Strict;
    check_numeric = p.Results.Numeric;
    
    result = struct();
    result.valid = true;
    result.warnings = {};
    result.errors = {};
    
    try
        % Basic structure validation
        if isstruct(data)
            fields = fieldnames(data);
            
            % Check for required fields
            required_fields = {'timestamp', 'parameters', 'results'};
            missing_fields = setdiff(required_fields, fields);
            
            if ~isempty(missing_fields) && strict
                result.valid = false;
                result.errors{end+1} = sprintf('Missing required fields: %s', ...
                                              strjoin(missing_fields, ', '));
            end
            
            % Check numeric data
            if check_numeric
                for i = 1:length(fields)
                    field_data = data.(fields{i});
                    if isnumeric(field_data)
                        % Check for NaN, Inf
                        if any(isnan(field_data(:)))
                            result.warnings{end+1} = sprintf('NaN values in field: %s', fields{i});
                        end
                        if any(isinf(field_data(:)))
                            result.warnings{end+1} = sprintf('Inf values in field: %s', fields{i});
                        end
                        if any(~isreal(field_data(:)))
                            result.warnings{end+1} = sprintf('Complex values in field: %s', fields{i});
                        end
                    end
                end
            end
        end
        
        % Schema validation if provided
        if ~isempty(fieldnames(schema))
            schema_result = validate_against_schema(data, schema);
            if ~schema_result.valid
                result.valid = false;
                result.errors = [result.errors, schema_result.errors];
            end
        end
        
        result.message = sprintf('Validation %s. %d errors, %d warnings.', ...
                               iif(result.valid, 'passed', 'failed'), ...
                               length(result.errors), length(result.warnings));
        
    catch ME
        result.valid = false;
        result.errors{end+1} = sprintf('Validation error: %s', ME.message);
        result.message = 'Validation failed due to error';
    end
end

function merged_data = merge_data(varargin)
    % Merge multiple datasets
    
    if nargin < 2
        error('At least two datasets required for merging');
    end
    
    datasets = varargin;
    
    % Check all inputs are structures
    for i = 1:length(datasets)
        if ~isstruct(datasets{i})
            error('All inputs must be structures');
        end
    end
    
    % Start with first dataset
    merged_data = datasets{1};
    
    % Merge remaining datasets
    for i = 2:length(datasets)
        current_data = datasets{i};
        fields = fieldnames(current_data);
        
        for j = 1:length(fields)
            field_name = fields{j};
            
            if isfield(merged_data, field_name)
                % Field exists - attempt to merge
                if isnumeric(merged_data.(field_name)) && isnumeric(current_data.(field_name))
                    % Concatenate numeric arrays
                    merged_data.(field_name) = [merged_data.(field_name); current_data.(field_name)];
                elseif iscell(merged_data.(field_name)) && iscell(current_data.(field_name))
                    % Concatenate cell arrays
                    merged_data.(field_name) = [merged_data.(field_name), current_data.(field_name)];
                else
                    % Create cell array for mixed types
                    merged_data.(field_name) = {merged_data.(field_name), current_data.(field_name)};
                end
            else
                % New field - add directly
                merged_data.(field_name) = current_data.(field_name);
            end
        end
    end
end

function extracted_data = extract_data(data, varargin)
    % Extract subset of data based on criteria
    
    p = inputParser;
    addRequired(p, 'data');
    addParameter(p, 'Fields', {}, @iscell);
    addParameter(p, 'Indices', [], @isnumeric);
    addParameter(p, 'Condition', [], @(x) isa(x, 'function_handle'));
    addParameter(p, 'Range', [], @isnumeric);
    
    parse(p, data, varargin{:});
    
    fields = p.Results.Fields;
    indices = p.Results.Indices;
    condition = p.Results.Condition;
    range = p.Results.Range;
    
    extracted_data = struct();
    
    if isstruct(data)
        % Extract specific fields
        if ~isempty(fields)
            for i = 1:length(fields)
                if isfield(data, fields{i})
                    extracted_data.(fields{i}) = data.(fields{i});
                end
            end
        else
            extracted_data = data;
        end
        
        % Apply indices/range/condition to numeric fields
        field_names = fieldnames(extracted_data);
        for i = 1:length(field_names)
            field_data = extracted_data.(field_names{i});
            
            if isnumeric(field_data)
                if ~isempty(indices)
                    field_data = field_data(indices, :);
                elseif ~isempty(range)
                    if length(range) == 2
                        field_data = field_data(range(1):range(2), :);
                    end
                elseif ~isempty(condition)
                    valid_idx = condition(field_data);
                    field_data = field_data(valid_idx, :);
                end
                
                extracted_data.(field_names{i}) = field_data;
            end
        end
    end
end

% Helper functions
function format = determine_format(ext)
    switch lower(ext)
        case '.mat'
            format = 'mat';
        case {'.json', '.js'}
            format = 'json';
        case {'.h5', '.hdf5'}
            format = 'hdf5';
        case {'.csv', '.txt'}
            format = 'csv';
        case {'.xls', '.xlsx'}
            format = 'excel';
        otherwise
            format = 'unknown';
    end
end

function version = get_spincirc_version()
    % Get SpinCirc version information
    version = '1.0.0';  % Default version
end

function checksum = calculate_checksum(data)
    % Calculate MD5 checksum of data
    try
        data_str = mat2str(data);
        checksum = matlab.internal.hash.MD5(data_str);
    catch
        checksum = 'unavailable';
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

% Format-specific save/load functions (simplified implementations)
function save_json(filename, data, metadata)
    combined = struct('data', data, 'metadata', metadata);
    json_str = jsonencode(combined);
    
    fid = fopen(filename, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
end

function [data, metadata] = load_json(filename)
    fid = fopen(filename, 'r');
    json_str = fread(fid, '*char')';
    fclose(fid);
    
    combined = jsondecode(json_str);
    data = combined.data;
    metadata = combined.metadata;
end

function export_csv(data, filename, precision, headers, delimiter)
    if isstruct(data)
        % Convert structure to table if possible
        try
            T = struct2table(data);
            writetable(T, filename, 'Delimiter', delimiter);
        catch
            error('Cannot convert structure to CSV format');
        end
    elseif isnumeric(data)
        dlmwrite(filename, data, 'delimiter', delimiter, 'precision', precision);
    else
        error('Unsupported data type for CSV export');
    end
end

function data = import_csv(filename, delimiter, headers)
    if strcmpi(delimiter, 'auto')
        data = readtable(filename);
    else
        data = readtable(filename, 'Delimiter', delimiter);
    end
end

function backup_filename = backup_data(filename)
    [path, name, ext] = fileparts(filename);
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    backup_filename = fullfile(path, [name, '_backup_', timestamp, ext]);
    copyfile(filename, backup_filename);
end

function success = batch_process(pattern, process_func, varargin)
    % Process multiple files matching pattern
    
    files = dir(pattern);
    success = true;
    
    for i = 1:length(files)
        try
            full_path = fullfile(files(i).folder, files(i).name);
            process_func(full_path, varargin{:});
        catch ME
            warning('Failed to process %s: %s', files(i).name, ME.message);
            success = false;
        end
    end
end

function result = validate_against_schema(data, schema)
    % Validate data against provided schema
    result = struct('valid', true, 'errors', {});
    
    % This is a simplified schema validation
    % In practice, this would be much more comprehensive
    schema_fields = fieldnames(schema);
    
    for i = 1:length(schema_fields)
        field_name = schema_fields{i};
        if ~isfield(data, field_name)
            result.valid = false;
            result.errors{end+1} = sprintf('Missing required field: %s', field_name);
        end
    end
end