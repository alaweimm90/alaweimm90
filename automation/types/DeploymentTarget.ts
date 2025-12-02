/**
 * Enumeration of supported deployment environments
 */
export enum DeploymentTarget {
  LOCAL = 'local',
  DEVELOPMENT = 'development',
  STAGING = 'staging',
  PRODUCTION = 'production',
  KUBERNETES = 'kubernetes',
  DOCKER = 'docker',
  SERVERLESS = 'serverless',
  EDGE = 'edge'
}

/**
 * Configuration for deployment targets
 */
export interface DeploymentConfig {
  target: DeploymentTarget;
  region?: string;
  environment?: Record<string, string>;
  resources?: ResourceRequirements;
  scaling?: ScalingConfig;
  network?: NetworkConfig;
  security?: SecurityConfig;
}

/**
 * Resource requirements for deployment
 */
export interface ResourceRequirements {
  cpu?: string;
  memory?: string;
  storage?: string;
  gpu?: string;
}

/**
 * Scaling configuration
 */
export interface ScalingConfig {
  minInstances?: number;
  maxInstances?: number;
  targetCpuUtilization?: number;
  targetMemoryUtilization?: number;
}

/**
 * Network configuration
 */
export interface NetworkConfig {
  ports?: number[];
  ingress?: boolean;
  loadBalancer?: boolean;
  domain?: string;
}

/**
 * Security configuration
 */
export interface SecurityConfig {
  secrets?: string[];
  certificates?: string[];
  authentication?: string[];
  authorization?: string[];
}
