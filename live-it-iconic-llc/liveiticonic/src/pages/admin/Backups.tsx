import { useState, useEffect } from 'react';
import {
  Download,
  Upload,
  RefreshCw,
  Trash2,
  AlertCircle,
  CheckCircle,
  Clock,
  HardDrive,
  Database,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { format } from 'date-fns';

interface BackupInfo {
  id: string;
  timestamp: string;
  tables: string[];
  recordCounts: Record<string, number>;
  totalRecords: number;
  size: number;
  status: 'success' | 'failed';
  createdAt: string;
}

interface BackupStats {
  totalBackups: number;
  totalSize: number;
  oldestBackup?: string;
  newestBackup?: string;
  averageBackupSize: number;
}

export function BackupsPage() {
  const [backups, setBackups] = useState<BackupInfo[]>([]);
  const [stats, setStats] = useState<BackupStats | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedBackup, setSelectedBackup] = useState<string | null>(null);
  const [showRestoreConfirm, setShowRestoreConfirm] = useState(false);

  useEffect(() => {
    loadBackups();
    loadStats();

    // Refresh every 30 seconds
    const interval = setInterval(() => {
      loadBackups();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const loadBackups = async () => {
    try {
      const response = await fetch('/api/admin/backups');
      const data = await response.json();

      if (data.success) {
        setBackups(data.data.backups);
        setError(null);
      } else {
        setError(data.error || 'Failed to load backups');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load backups');
    } finally {
      setIsLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch('/api/admin/backups/stats');
      const data = await response.json();

      if (data.success) {
        setStats(data.data);
      }
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  };

  const createBackup = async () => {
    setIsCreating(true);
    setError(null);

    try {
      const response = await fetch('/api/admin/backups', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      const data = await response.json();

      if (data.success) {
        loadBackups();
        loadStats();
      } else {
        setError(data.error || 'Failed to create backup');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create backup');
    } finally {
      setIsCreating(false);
    }
  };

  const deleteBackup = async (id: string) => {
    if (!confirm('Are you sure you want to delete this backup?')) {
      return;
    }

    try {
      const response = await fetch(`/api/admin/backups/${id}`, {
        method: 'DELETE',
      });

      const data = await response.json();

      if (data.success) {
        loadBackups();
        loadStats();
      } else {
        setError(data.error || 'Failed to delete backup');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete backup');
    }
  };

  const restoreBackup = async (id: string) => {
    try {
      const response = await fetch(`/api/admin/backups/${id}/restore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true }),
      });

      const data = await response.json();

      if (data.success) {
        setShowRestoreConfirm(false);
        setSelectedBackup(null);
      } else {
        setError(data.message || 'Failed to restore backup');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to restore backup');
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Database Backups</h1>
            <p className="text-slate-600 mt-2">Manage and monitor database backups</p>
          </div>
          <Button
            onClick={createBackup}
            disabled={isCreating}
            className="flex items-center gap-2"
          >
            {isCreating ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <RefreshCw className="w-4 h-4" />
                Create Backup
              </>
            )}
          </Button>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-gap gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
            <div>
              <p className="font-semibold text-red-900">Error</p>
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        )}

        {/* Statistics Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">Total Backups</p>
                  <p className="text-2xl font-bold text-slate-900">{stats.totalBackups}</p>
                </div>
                <Database className="w-8 h-8 text-blue-500 opacity-20" />
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">Total Size</p>
                  <p className="text-2xl font-bold text-slate-900">
                    {formatBytes(stats.totalSize)}
                  </p>
                </div>
                <HardDrive className="w-8 h-8 text-green-500 opacity-20" />
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">Avg Size</p>
                  <p className="text-2xl font-bold text-slate-900">
                    {formatBytes(stats.averageBackupSize)}
                  </p>
                </div>
                <Database className="w-8 h-8 text-purple-500 opacity-20" />
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">Latest</p>
                  <p className="text-sm font-mono text-slate-900">
                    {stats.newestBackup
                      ? format(new Date(stats.newestBackup), 'MMM dd')
                      : 'Never'}
                  </p>
                </div>
                <Clock className="w-8 h-8 text-orange-500 opacity-20" />
              </div>
            </div>
          </div>
        )}

        {/* Backups Table */}
        <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
          {isLoading ? (
            <div className="p-8 text-center">
              <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2 text-slate-400" />
              <p className="text-slate-600">Loading backups...</p>
            </div>
          ) : backups.length === 0 ? (
            <div className="p-8 text-center">
              <Database className="w-12 h-12 text-slate-300 mx-auto mb-2" />
              <p className="text-slate-600">No backups yet</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-slate-900">
                      Date
                    </th>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-slate-900">
                      Tables
                    </th>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-slate-900">
                      Records
                    </th>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-slate-900">
                      Size
                    </th>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-slate-900">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-slate-900">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200">
                  {backups.map((backup) => (
                    <tr key={backup.id} className="hover:bg-slate-50">
                      <td className="px-6 py-4 text-sm">
                        <div className="font-medium text-slate-900">
                          {format(new Date(backup.timestamp), 'MMM dd, yyyy HH:mm')}
                        </div>
                        <div className="text-xs text-slate-500">
                          {format(new Date(backup.timestamp), 'zzz')}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        <div className="flex flex-wrap gap-1">
                          {backup.tables.map((table) => (
                            <span
                              key={table}
                              className="inline-block bg-slate-100 text-slate-700 px-2 py-1 rounded text-xs"
                            >
                              {table}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-900 font-semibold">
                        {backup.totalRecords.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        {formatBytes(backup.size)}
                      </td>
                      <td className="px-6 py-4 text-sm">
                        <div className="flex items-center gap-2">
                          {backup.status === 'success' ? (
                            <>
                              <CheckCircle className="w-4 h-4 text-green-600" />
                              <span className="text-green-700">Success</span>
                            </>
                          ) : (
                            <>
                              <AlertCircle className="w-4 h-4 text-red-600" />
                              <span className="text-red-700">Failed</span>
                            </>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm">
                        <div className="flex items-center gap-2">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => {
                              setSelectedBackup(backup.id);
                              setShowRestoreConfirm(true);
                            }}
                            title="Restore from this backup"
                          >
                            <Upload className="w-4 h-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => deleteBackup(backup.id)}
                            title="Delete this backup"
                          >
                            <Trash2 className="w-4 h-4 text-red-500" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Restore Confirmation Dialog */}
        {showRestoreConfirm && selectedBackup && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg max-w-md w-full p-6">
              <h2 className="text-xl font-bold text-slate-900 mb-4">Restore Database?</h2>
              <p className="text-slate-600 mb-6">
                This will replace all current data with the selected backup. This action cannot be
                undone without another backup.
              </p>
              <div className="bg-red-50 border border-red-200 rounded p-3 mb-6">
                <p className="text-sm text-red-700">
                  <strong>Warning:</strong> Make sure you have a recent backup before proceeding.
                </p>
              </div>
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  onClick={() => setShowRestoreConfirm(false)}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  onClick={() => restoreBackup(selectedBackup)}
                  className="flex-1 bg-red-600 hover:bg-red-700"
                >
                  Restore
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Info Section */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900 mb-2">About Backups</h3>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Automated backups run daily at 2:00 AM UTC</li>
            <li>• Local backups are retained for 30 days</li>
            <li>• S3 backups are retained for 90 days</li>
            <li>• All backups include metadata and record counts</li>
            <li>• Recovery Time Objective (RTO): 1 hour for critical services</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default BackupsPage;
