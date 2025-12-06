#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('📊 Backup Health Monitor\n');

const monitoringReport = {
  timestamp: new Date().toISOString(),
  backupHealth: {
    database: 'unknown',
    files: 'unknown',
    config: 'unknown'
  },
  alerts: [],
  recommendations: []
};

function checkBackupHealth() {
  const backupDir = path.join(rootDir, 'backups');
  
  if (!fs.existsSync(backupDir)) {
    monitoringReport.alerts.push({
      level: 'critical',
      message: 'Backup directory does not exist',
      action: 'Run backup setup script'
    });
    return;
  }
  
  // Check database backups
  const dbBackupDir = path.join(backupDir, 'database');
  if (fs.existsSync(dbBackupDir)) {
    const dbBackups = fs.readdirSync(dbBackupDir).filter(f => f.endsWith('.sql.gz'));
    if (dbBackups.length > 0) {
      const latestBackup = dbBackups.sort().pop();
      const backupPath = path.join(dbBackupDir, latestBackup);
      const stat = fs.statSync(backupPath);
      const ageHours = (Date.now() - stat.mtime) / (1000 * 60 * 60);
      
      if (ageHours < 24) {
        monitoringReport.backupHealth.database = 'healthy';
        console.log('✅ Database backup is current');
      } else if (ageHours < 48) {
        monitoringReport.backupHealth.database = 'warning';
        monitoringReport.alerts.push({
          level: 'warning',
          message: `Database backup is ${Math.round(ageHours)} hours old`,
          action: 'Check backup schedule'
        });
      } else {
        monitoringReport.backupHealth.database = 'critical';
        monitoringReport.alerts.push({
          level: 'critical',
          message: `Database backup is ${Math.round(ageHours)} hours old`,
          action: 'Run database backup immediately'
        });
      }
    } else {
      monitoringReport.backupHealth.database = 'critical';
      monitoringReport.alerts.push({
        level: 'critical',
        message: 'No database backups found',
        action: 'Run database backup immediately'
      });
    }
  }
  
  // Check file backups
  const fileBackupDir = path.join(backupDir, 'files');
  if (fs.existsSync(fileBackupDir)) {
    const fileBackups = fs.readdirSync(fileBackupDir).filter(f => f.endsWith('.tar.gz'));
    if (fileBackups.length > 0) {
      monitoringReport.backupHealth.files = 'healthy';
      console.log('✅ File backups exist');
    } else {
      monitoringReport.backupHealth.files = 'warning';
      monitoringReport.alerts.push({
        level: 'warning',
        message: 'No file backups found',
        action: 'Run file backup script'
      });
    }
  }
  
  // Check config backups
  const configBackupDir = path.join(backupDir, 'config');
  if (fs.existsSync(configBackupDir)) {
    const configBackups = fs.readdirSync(configBackupDir).filter(f => f.endsWith('.tar.gz'));
    if (configBackups.length > 0) {
      monitoringReport.backupHealth.config = 'healthy';
      console.log('✅ Configuration backups exist');
    } else {
      monitoringReport.backupHealth.config = 'warning';
      monitoringReport.alerts.push({
        level: 'warning',
        message: 'No configuration backups found',
        action: 'Run configuration backup script'
      });
    }
  }
}

function generateRecommendations() {
  if (monitoringReport.alerts.length === 0) {
    console.log('\n✅ All backup systems healthy');
  } else {
    console.log('\n🚨 Backup issues detected:');
    monitoringReport.alerts.forEach(alert => {
      const icon = alert.level === 'critical' ? '🔴' : '⚠️';
      console.log(`  ${icon} ${alert.message}`);
    });
  }
  
  monitoringReport.recommendations = [
    'Set up automated daily backups',
    'Test disaster recovery procedures monthly',
    'Monitor backup sizes and integrity',
    'Implement off-site backup storage',
    'Document recovery procedures'
  ];
}

checkBackupHealth();
generateRecommendations();

// Save monitoring report
fs.writeFileSync(
  path.join(rootDir, 'backup-monitoring-report.json'),
  JSON.stringify(monitoringReport, null, 2)
);

console.log('\n📄 Backup monitoring report saved');
