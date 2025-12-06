/**
 * Publication Timeline Component
 * Interactive timeline visualization for research papers
 */

import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
  ReferenceLine,
  Dot,
} from 'recharts';
import { Card } from '../../../components/Card';
import { Badge } from '../../../components/Badge';
import { Button } from '../../../components/Button';
import { Select } from '../../../components/Select';
import { Calendar, TrendingUp, Award, Users } from 'lucide-react';

export interface PublicationData {
  year: number;
  month?: number;
  publications: number;
  citations: number;
  collaborators: number;
  impactFactor?: number;
  topics?: string[];
  keyPapers?: {
    id: string;
    title: string;
    citations: number;
  }[];
}

export interface PublicationTimelineProps {
  data: PublicationData[];
  selectedYear?: number;
  onYearSelect?: (year: number) => void;
  className?: string;
  height?: number;
}

export const PublicationTimeline: React.FC<PublicationTimelineProps> = ({
  data,
  selectedYear,
  onYearSelect,
  className = '',
  height = 400,
}) => {
  const [viewType, setViewType] = React.useState<'area' | 'bar' | 'line'>('area');
  const [metric, setMetric] = React.useState<'publications' | 'citations' | 'impact'>('publications');
  const [timeRange, setTimeRange] = React.useState<'all' | '5y' | '10y'>('all');

  // Process and filter data based on time range
  const processedData = useMemo(() => {
    let filteredData = [...data];

    // Apply time range filter
    if (timeRange !== 'all') {
      const currentYear = new Date().getFullYear();
      const yearsToShow = timeRange === '5y' ? 5 : 10;
      const startYear = currentYear - yearsToShow;
      filteredData = filteredData.filter((d) => d.year >= startYear);
    }

    // Sort by year
    filteredData.sort((a, b) => a.year - b.year);

    // Calculate additional metrics
    return filteredData.map((item) => ({
      ...item,
      impactScore: item.citations / Math.max(item.publications, 1),
      avgCitations: item.citations / Math.max(item.publications, 1),
    }));
  }, [data, timeRange]);

  // Calculate statistics
  const stats = useMemo(() => {
    const totalPublications = processedData.reduce((sum, d) => sum + d.publications, 0);
    const totalCitations = processedData.reduce((sum, d) => sum + d.citations, 0);
    const avgCollaborators =
      processedData.reduce((sum, d) => sum + d.collaborators, 0) / processedData.length;
    const peakYear = processedData.reduce((max, d) =>
      d.publications > max.publications ? d : max, processedData[0]);

    return {
      totalPublications,
      totalCitations,
      avgCollaborators: Math.round(avgCollaborators),
      peakYear: peakYear?.year,
    };
  }, [processedData]);

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload[0]) return null;

    const data = payload[0].payload;

    return (
      <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
        <p className="font-semibold text-sm text-gray-900 dark:text-white mb-2">
          {label}
        </p>
        <div className="space-y-1">
          <div className="flex justify-between items-center space-x-4">
            <span className="text-xs text-gray-600 dark:text-gray-400">Publications:</span>
            <span className="text-xs font-medium text-gray-900 dark:text-white">
              {data.publications}
            </span>
          </div>
          <div className="flex justify-between items-center space-x-4">
            <span className="text-xs text-gray-600 dark:text-gray-400">Citations:</span>
            <span className="text-xs font-medium text-gray-900 dark:text-white">
              {data.citations}
            </span>
          </div>
          <div className="flex justify-between items-center space-x-4">
            <span className="text-xs text-gray-600 dark:text-gray-400">Collaborators:</span>
            <span className="text-xs font-medium text-gray-900 dark:text-white">
              {data.collaborators}
            </span>
          </div>
          {data.impactFactor && (
            <div className="flex justify-between items-center space-x-4">
              <span className="text-xs text-gray-600 dark:text-gray-400">Impact Factor:</span>
              <span className="text-xs font-medium text-gray-900 dark:text-white">
                {data.impactFactor.toFixed(2)}
              </span>
            </div>
          )}
        </div>
        {data.keyPapers && data.keyPapers.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
            <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
              Key Papers:
            </p>
            {data.keyPapers.slice(0, 2).map((paper: any) => (
              <p key={paper.id} className="text-xs text-gray-600 dark:text-gray-400 truncate">
                • {paper.title}
              </p>
            ))}
          </div>
        )}
      </div>
    );
  };

  // Render chart based on view type
  const renderChart = () => {
    const commonProps = {
      data: processedData,
      margin: { top: 10, right: 30, left: 0, bottom: 0 },
    };

    const getDataKey = () => {
      switch (metric) {
        case 'citations':
          return 'citations';
        case 'impact':
          return 'impactScore';
        default:
          return 'publications';
      }
    };

    switch (viewType) {
      case 'area':
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="year"
              stroke="#6b7280"
              tick={{ fontSize: 12 }}
              onClick={(data) => data && onYearSelect?.(data.value)}
            />
            <YAxis
              stroke="#6b7280"
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Area
              type="monotone"
              dataKey={getDataKey()}
              stroke="#3b82f6"
              fill="#3b82f6"
              fillOpacity={0.3}
              strokeWidth={2}
              animationDuration={1000}
            />
            {metric === 'publications' && (
              <Area
                type="monotone"
                dataKey="collaborators"
                stroke="#10b981"
                fill="#10b981"
                fillOpacity={0.2}
                strokeWidth={2}
              />
            )}
            {selectedYear && (
              <ReferenceLine
                x={selectedYear}
                stroke="#f59e0b"
                strokeWidth={2}
                label={{ value: 'Selected', position: 'top', fill: '#f59e0b' }}
              />
            )}
            <Brush
              dataKey="year"
              height={30}
              stroke="#3b82f6"
              fill="#eff6ff"
            />
          </AreaChart>
        );

      case 'bar':
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="year"
              stroke="#6b7280"
              tick={{ fontSize: 12 }}
            />
            <YAxis
              stroke="#6b7280"
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar
              dataKey={getDataKey()}
              fill="#3b82f6"
              radius={[4, 4, 0, 0]}
              animationDuration={1000}
            />
            {metric === 'publications' && (
              <Bar
                dataKey="citations"
                fill="#10b981"
                radius={[4, 4, 0, 0]}
              />
            )}
            {selectedYear && (
              <ReferenceLine
                x={selectedYear}
                stroke="#f59e0b"
                strokeWidth={2}
              />
            )}
          </BarChart>
        );

      case 'line':
      default:
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="year"
              stroke="#6b7280"
              tick={{ fontSize: 12 }}
            />
            <YAxis
              stroke="#6b7280"
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey={getDataKey()}
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ fill: '#3b82f6', r: 4 }}
              activeDot={{ r: 6 }}
              animationDuration={1000}
            />
            {metric === 'publications' && (
              <>
                <Line
                  type="monotone"
                  dataKey="citations"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={{ fill: '#10b981', r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="collaborators"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ fill: '#f59e0b', r: 3 }}
                />
              </>
            )}
            {selectedYear && (
              <ReferenceLine
                x={selectedYear}
                stroke="#ef4444"
                strokeWidth={2}
              />
            )}
          </LineChart>
        );
    }
  };

  return (
    <Card className={className}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Publication Timeline
          </h3>

          <div className="flex items-center space-x-2">
            {/* View Type Selector */}
            <div className="flex items-center space-x-1">
              <Button
                size="sm"
                variant={viewType === 'area' ? 'primary' : 'ghost'}
                onClick={() => setViewType('area')}
              >
                Area
              </Button>
              <Button
                size="sm"
                variant={viewType === 'bar' ? 'primary' : 'ghost'}
                onClick={() => setViewType('bar')}
              >
                Bar
              </Button>
              <Button
                size="sm"
                variant={viewType === 'line' ? 'primary' : 'ghost'}
                onClick={() => setViewType('line')}
              >
                Line
              </Button>
            </div>

            {/* Metric Selector */}
            <Select
              value={metric}
              onChange={(e) => setMetric(e.target.value as any)}
              className="text-sm"
            >
              <option value="publications">Publications</option>
              <option value="citations">Citations</option>
              <option value="impact">Impact Score</option>
            </Select>

            {/* Time Range Selector */}
            <Select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as any)}
              className="text-sm"
            >
              <option value="all">All Time</option>
              <option value="10y">Last 10 Years</option>
              <option value="5y">Last 5 Years</option>
            </Select>
          </div>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-4 gap-4">
          <div className="flex items-center space-x-2">
            <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <Calendar className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-xs text-gray-600 dark:text-gray-400">Total Papers</p>
              <p className="text-sm font-semibold text-gray-900 dark:text-white">
                {stats.totalPublications}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <Award className="w-4 h-4 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-xs text-gray-600 dark:text-gray-400">Citations</p>
              <p className="text-sm font-semibold text-gray-900 dark:text-white">
                {stats.totalCitations}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <div className="p-2 bg-amber-100 dark:bg-amber-900/30 rounded-lg">
              <Users className="w-4 h-4 text-amber-600 dark:text-amber-400" />
            </div>
            <div>
              <p className="text-xs text-gray-600 dark:text-gray-400">Avg Collaborators</p>
              <p className="text-sm font-semibold text-gray-900 dark:text-white">
                {stats.avgCollaborators}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <TrendingUp className="w-4 h-4 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <p className="text-xs text-gray-600 dark:text-gray-400">Peak Year</p>
              <p className="text-sm font-semibold text-gray-900 dark:text-white">
                {stats.peakYear}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="p-4">
        <ResponsiveContainer width="100%" height={height}>
          {renderChart()}
        </ResponsiveContainer>
      </div>
    </Card>
  );
};

export default PublicationTimeline;