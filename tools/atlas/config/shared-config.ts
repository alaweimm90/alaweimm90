export interface SharedConfig {
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  metricsPath: string;
}

export const sharedConfig: SharedConfig = {
  logLevel: 'info',
  metricsPath: '.atlas/metrics.json',
};

export default sharedConfig;
