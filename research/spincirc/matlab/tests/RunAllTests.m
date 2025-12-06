function results = RunAllTests(varargin)
% RUNALLTESTS - Comprehensive test suite runner for SpinCirc
%
% This function runs all unit tests in the SpinCirc MATLAB codebase and
% provides detailed reporting on test results, coverage, and performance.
%
% Syntax:
%   results = RunAllTests()
%   results = RunAllTests('option', value, ...)
%
% Options:
%   'Verbose'     - Display detailed test output (default: true)
%   'Coverage'    - Generate code coverage report (default: false)
%   'Profile'     - Profile test execution (default: false)
%   'Parallel'    - Run tests in parallel (default: false)
%   'OutputFile'  - Save results to file (default: '')
%   'TestFilter'  - Filter tests by name pattern (default: '')
%   'StopOnFail'  - Stop on first failure (default: false)
%   'Timeout'     - Test timeout in seconds (default: 300)
%
% Returns:
%   results - TestResult object with detailed test information
%
% Examples:
%   % Run all tests with default settings
%   results = RunAllTests()
%   
%   % Run with coverage and save to file
%   results = RunAllTests('Coverage', true, 'OutputFile', 'test_results.xml')
%   
%   % Run only unit conversion tests
%   results = RunAllTests('TestFilter', '*UnitConversions*')
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'Verbose', true, @islogical);
    addParameter(p, 'Coverage', false, @islogical);
    addParameter(p, 'Profile', false, @islogical);
    addParameter(p, 'Parallel', false, @islogical);
    addParameter(p, 'OutputFile', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'TestFilter', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'StopOnFail', false, @islogical);
    addParameter(p, 'Timeout', 300, @(x) isnumeric(x) && isscalar(x) && x > 0);
    
    parse(p, varargin{:});
    
    verbose = p.Results.Verbose;
    coverage = p.Results.Coverage;
    profile_tests = p.Results.Profile;
    parallel = p.Results.Parallel;
    output_file = p.Results.OutputFile;
    test_filter = p.Results.TestFilter;
    stop_on_fail = p.Results.StopOnFail;
    timeout = p.Results.Timeout;
    
    % Initialize test environment
    fprintf('Initializing SpinCirc Test Suite\n');
    fprintf('================================\n\n');
    
    % Get test directory
    test_dir = fileparts(mfilename('fullpath'));
    core_dir = fullfile(test_dir, '..', 'core');
    
    % Add necessary paths
    addpath(genpath(core_dir));
    cleanup_paths = onCleanup(@() rmpath(genpath(core_dir)));
    
    % Create test suite
    try
        suite = create_test_suite(test_dir, test_filter);
    catch ME
        error('Failed to create test suite: %s', ME.message);
    end
    
    if isempty(suite)
        warning('No tests found matching filter: %s', test_filter);
        results = [];
        return;
    end
    
    fprintf('Found %d test classes with %d total tests\n', ...
           length(unique({suite.TestClass})), length(suite));
    
    % Configure test runner
    runner = create_test_runner(verbose, parallel, stop_on_fail, timeout);
    
    % Start profiling if requested
    if profile_tests
        profile('on');
        profile_cleanup = onCleanup(@() profile('off'));
    end
    
    % Start coverage if requested
    if coverage
        coverage_cleanup = start_coverage_collection(core_dir);
    end
    
    % Run tests
    fprintf('\nRunning Tests\n');
    fprintf('=============\n\n');
    
    start_time = tic;
    
    try
        results = run(runner, suite);
    catch ME
        error('Test execution failed: %s', ME.message);
    end
    
    total_time = toc(start_time);
    
    % Process results
    fprintf('\nTest Results Summary\n');
    fprintf('===================\n');
    
    display_test_summary(results, total_time);
    
    % Generate detailed report
    if verbose
        display_detailed_results(results);
    end
    
    % Save results to file if requested
    if ~isempty(output_file)
        save_test_results(results, output_file);
        fprintf('Results saved to: %s\n', output_file);
    end
    
    % Generate coverage report
    if coverage
        generate_coverage_report(core_dir);
    end
    
    % Generate profiling report
    if profile_tests
        generate_profiling_report();
    end
    
    % Display recommendations
    display_recommendations(results);
    
    % Return summary information
    fprintf('\n');
    if results.NumPassed == length(results.Details)
        fprintf('🎉 ALL TESTS PASSED! 🎉\n');
    else
        fprintf('❌ Some tests failed. Review the detailed results above.\n');
    end
    
    fprintf('Run completed in %.2f seconds\n', total_time);
end

function suite = create_test_suite(test_dir, test_filter)
    % Create test suite from test directory
    
    if isempty(test_filter)
        suite = testsuite(test_dir);
    else
        % Find matching test files
        test_files = dir(fullfile(test_dir, 'Test*.m'));
        matching_files = {};
        
        for i = 1:length(test_files)
            if contains(test_files(i).name, test_filter, 'IgnoreCase', true)
                [~, class_name, ~] = fileparts(test_files(i).name);
                matching_files{end+1} = class_name;
            end
        end
        
        if isempty(matching_files)
            suite = [];
        else
            suite = testsuite(matching_files);
        end
    end
end

function runner = create_test_runner(verbose, parallel, stop_on_fail, timeout)
    % Create and configure test runner
    
    import matlab.unittest.TestRunner;
    import matlab.unittest.plugins.TestReportPlugin;
    import matlab.unittest.plugins.StopOnFailuresPlugin;
    import matlab.unittest.plugins.TimeoutPlugin;
    
    % Create basic runner
    if parallel && exist('parpool', 'file') == 2
        try
            if isempty(gcp('nocreate'))
                parpool('local');
            end
            runner = TestRunner.withNoPlugins();
            fprintf('Running tests in parallel mode\n');
        catch
            fprintf('Parallel execution failed, falling back to serial\n');
            runner = TestRunner.withTextOutput();
        end
    else
        if verbose
            runner = TestRunner.withTextOutput();
        else
            runner = TestRunner.withNoPlugins();
        end
    end
    
    % Add plugins
    if stop_on_fail
        runner.addPlugin(StopOnFailuresPlugin());
    end
    
    if timeout > 0
        runner.addPlugin(TimeoutPlugin.forDuration(seconds(timeout)));
    end
end

function display_test_summary(results, total_time)
    % Display summary of test results
    
    total_tests = length(results.Details);
    passed = results.NumPassed;
    failed = results.NumFailed;
    incomplete = results.NumIncomplete;
    
    fprintf('Total Tests:     %d\n', total_tests);
    fprintf('Passed:          %d (%.1f%%)\n', passed, 100*passed/total_tests);
    fprintf('Failed:          %d (%.1f%%)\n', failed, 100*failed/total_tests);
    fprintf('Incomplete:      %d (%.1f%%)\n', incomplete, 100*incomplete/total_tests);
    fprintf('Total Time:      %.2f seconds\n', total_time);
    
    if total_tests > 0
        fprintf('Average per test: %.3f seconds\n', total_time/total_tests);
    end
end

function display_detailed_results(results)
    % Display detailed test results
    
    fprintf('\nDetailed Results\n');
    fprintf('================\n');
    
    % Group results by test class
    test_classes = unique({results.Details.TestClass});
    
    for i = 1:length(test_classes)
        class_name = test_classes{i};
        class_tests = results.Details(strcmp({results.Details.TestClass}, class_name));
        
        fprintf('\n%s:\n', class_name);
        fprintf('%s\n', repmat('-', 1, length(class_name)+1));
        
        for j = 1:length(class_tests)
            test = class_tests(j);
            status = get_test_status(test);
            duration = test.Duration;
            
            fprintf('  %-40s %s (%.3fs)\n', test.Name, status, duration);
            
            % Show failure details
            if ~test.Passed && ~isempty(test.Details)
                if iscell(test.Details)
                    for k = 1:length(test.Details)
                        if ~isempty(test.Details{k}.Exception)
                            fprintf('    Error: %s\n', test.Details{k}.Exception.message);
                        end
                    end
                elseif ~isempty(test.Details.Exception)
                    fprintf('    Error: %s\n', test.Details.Exception.message);
                end
            end
        end
        
        % Class summary
        class_passed = sum([class_tests.Passed]);
        class_total = length(class_tests);
        fprintf('  Summary: %d/%d passed (%.1f%%)\n', ...
               class_passed, class_total, 100*class_passed/class_total);
    end
end

function status = get_test_status(test)
    % Get readable test status
    
    if test.Passed
        status = '✅ PASS';
    elseif test.Failed
        status = '❌ FAIL';
    elseif test.Incomplete
        status = '⚠️  INCOMPLETE';
    else
        status = '❓ UNKNOWN';
    end
end

function save_test_results(results, output_file)
    % Save test results to file
    
    [~, ~, ext] = fileparts(output_file);
    
    switch lower(ext)
        case '.xml'
            save_xml_results(results, output_file);
        case '.json'
            save_json_results(results, output_file);
        case '.mat'
            save(output_file, 'results');
        otherwise
            save_text_results(results, output_file);
    end
end

function save_xml_results(results, filename)
    % Save results in XML format (simplified)
    
    fid = fopen(filename, 'w');
    
    fprintf(fid, '<?xml version="1.0" encoding="UTF-8"?>\n');
    fprintf(fid, '<testsuites>\n');
    
    % Group by test class
    test_classes = unique({results.Details.TestClass});
    
    for i = 1:length(test_classes)
        class_name = test_classes{i};
        class_tests = results.Details(strcmp({results.Details.TestClass}, class_name));
        
        class_passed = sum([class_tests.Passed]);
        class_failed = sum([class_tests.Failed]);
        class_time = sum([class_tests.Duration]);
        
        fprintf(fid, '  <testsuite name="%s" tests="%d" failures="%d" time="%.3f">\n', ...
               class_name, length(class_tests), class_failed, class_time);
        
        for j = 1:length(class_tests)
            test = class_tests(j);
            fprintf(fid, '    <testcase name="%s" time="%.3f"', test.Name, test.Duration);
            
            if test.Failed
                fprintf(fid, '>\n      <failure>Test failed</failure>\n    </testcase>\n');
            else
                fprintf(fid, '/>\n');
            end
        end
        
        fprintf(fid, '  </testsuite>\n');
    end
    
    fprintf(fid, '</testsuites>\n');
    fclose(fid);
end

function save_json_results(results, filename)
    % Save results in JSON format
    
    result_struct = struct();
    result_struct.summary.total = length(results.Details);
    result_struct.summary.passed = results.NumPassed;
    result_struct.summary.failed = results.NumFailed;
    result_struct.summary.incomplete = results.NumIncomplete;
    
    result_struct.tests = results.Details;
    
    json_str = jsonencode(result_struct);
    
    fid = fopen(filename, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
end

function save_text_results(results, filename)
    % Save results in plain text format
    
    fid = fopen(filename, 'w');
    
    fprintf(fid, 'SpinCirc Test Results\n');
    fprintf(fid, '====================\n\n');
    
    fprintf(fid, 'Summary:\n');
    fprintf(fid, 'Total Tests: %d\n', length(results.Details));
    fprintf(fid, 'Passed: %d\n', results.NumPassed);
    fprintf(fid, 'Failed: %d\n', results.NumFailed);
    fprintf(fid, 'Incomplete: %d\n\n', results.NumIncomplete);
    
    fprintf(fid, 'Detailed Results:\n');
    for i = 1:length(results.Details)
        test = results.Details(i);
        status = iif(test.Passed, 'PASS', 'FAIL');
        fprintf(fid, '%s: %s (%.3fs)\n', test.Name, status, test.Duration);
    end
    
    fclose(fid);
end

function cleanup = start_coverage_collection(core_dir)
    % Start code coverage collection
    
    try
        if exist('codecov', 'file') == 2
            codecov('start', core_dir);
            cleanup = onCleanup(@() codecov('stop'));
            fprintf('Code coverage collection started\n');
        else
            fprintf('Code coverage tool not available\n');
            cleanup = [];
        end
    catch ME
        warning('Failed to start coverage collection: %s', ME.message);
        cleanup = [];
    end
end

function generate_coverage_report(core_dir)
    % Generate code coverage report
    
    try
        if exist('codecov', 'file') == 2
            report = codecov('report');
            fprintf('\nCode Coverage Report\n');
            fprintf('===================\n');
            fprintf('Overall Coverage: %.1f%%\n', report.coverage);
            
            % Display file-by-file coverage
            if isfield(report, 'files')
                fprintf('\nFile Coverage:\n');
                for i = 1:length(report.files)
                    file_info = report.files(i);
                    fprintf('  %-40s %.1f%%\n', file_info.name, file_info.coverage);
                end
            end
        end
    catch ME
        warning('Failed to generate coverage report: %s', ME.message);
    end
end

function generate_profiling_report()
    % Generate profiling report
    
    try
        profile_info = profile('info');
        
        if ~isempty(profile_info.FunctionTable)
            fprintf('\nProfiling Report\n');
            fprintf('===============\n');
            
            % Sort by total time
            [~, sort_idx] = sort([profile_info.FunctionTable.TotalTime], 'descend');
            top_functions = profile_info.FunctionTable(sort_idx(1:min(10, end)));
            
            fprintf('Top 10 functions by execution time:\n');
            for i = 1:length(top_functions)
                func = top_functions(i);
                fprintf('  %-30s %.3fs (%d calls)\n', ...
                       func.FunctionName, func.TotalTime, func.NumCalls);
            end
            
            % Generate HTML report
            profview(0, profile_info);
            fprintf('Detailed profiling report generated in browser\n');
        end
    catch ME
        warning('Failed to generate profiling report: %s', ME.message);
    end
end

function display_recommendations(results)
    % Display recommendations based on test results
    
    fprintf('\nRecommendations\n');
    fprintf('===============\n');
    
    if results.NumFailed > 0
        fprintf('• Address failing tests before deploying code\n');
        fprintf('• Review error messages and stack traces above\n');
    end
    
    if results.NumIncomplete > 0
        fprintf('• Investigate incomplete tests - they may indicate setup issues\n');
    end
    
    % Performance recommendations
    slow_tests = results.Details([results.Details.Duration] > 1.0);
    if ~isempty(slow_tests)
        fprintf('• Consider optimizing slow tests (>1s):\n');
        for i = 1:min(5, length(slow_tests))
            fprintf('  - %s (%.2fs)\n', slow_tests(i).Name, slow_tests(i).Duration);
        end
    end
    
    % General recommendations
    fprintf('• Run tests regularly during development\n');
    fprintf('• Add tests for new functionality\n');
    fprintf('• Consider adding performance benchmarks\n');
end

function result = iif(condition, true_value, false_value)
    % Inline if function
    if condition
        result = true_value;
    else
        result = false_value;
    end
end