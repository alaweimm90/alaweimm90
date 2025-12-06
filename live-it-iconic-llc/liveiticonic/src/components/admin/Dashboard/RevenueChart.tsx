/**
 * Revenue Chart Component
 * Displays revenue trend over time
 */
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import React, { useEffect, useRef, useState } from 'react';

const data = [
  { month: 'Jan', revenue: 4000, orders: 120 },
  { month: 'Feb', revenue: 5200, orders: 145 },
  { month: 'Mar', revenue: 4800, orders: 138 },
  { month: 'Apr', revenue: 6100, orders: 165 },
  { month: 'May', revenue: 7200, orders: 189 },
  { month: 'Jun', revenue: 8500, orders: 212 },
];

export function RevenueChart() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [inView, setInView] = useState(false);
  type RechartsSubset = Pick<
    typeof import('recharts'),
    'ResponsiveContainer' | 'LineChart' | 'Line' | 'CartesianGrid' | 'XAxis' | 'YAxis' | 'Tooltip'
  >;
  const [recharts, setRecharts] = useState<RechartsSubset | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting) {
          setInView(true);
          io.disconnect();
        }
      },
      { rootMargin: '200px' }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!inView || recharts) return;
    import('recharts').then(mod => {
      setRecharts({
        ResponsiveContainer: mod.ResponsiveContainer,
        LineChart: mod.LineChart,
        Line: mod.Line,
        CartesianGrid: mod.CartesianGrid,
        XAxis: mod.XAxis,
        YAxis: mod.YAxis,
        Tooltip: mod.Tooltip,
      });
    });
  }, [inView, recharts]);

  return (
    <Card className="bg-white border-lii-cloud">
      <CardHeader>
        <CardTitle className="text-lii-ink">Revenue Trend</CardTitle>
        <p className="text-sm text-lii-ash mt-1">Last 6 months performance</p>
      </CardHeader>
      <CardContent>
        <div ref={containerRef} className="h-80 w-full">
          {!recharts ? (
            <div className="h-full w-full bg-lii-cloud/20 animate-pulse rounded-lg" />
          ) : (
            <recharts.ResponsiveContainer width="100%" height="100%">
              <recharts.LineChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <recharts.CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <recharts.XAxis dataKey="month" stroke="#666" />
                <recharts.YAxis stroke="#666" />
                <recharts.Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                  }}
                />
                <recharts.Line
                  type="monotone"
                  dataKey="revenue"
                  stroke="#D4A574"
                  strokeWidth={2}
                  dot={{ fill: '#D4A574', r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </recharts.LineChart>
            </recharts.ResponsiveContainer>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
