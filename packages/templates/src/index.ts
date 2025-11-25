/**  
 * @monorepo/templates  
 *  
 * Consolidated templates package for issues and workflows  
 */  
  
// Issue Templates  
export * from './types';  
export { IssueManager, createDefaultTemplates as createDefaultIssueTemplates } from './issue-manager';  
  
// Workflow Templates  
export { WorkflowManager, createDefaultTemplates as createDefaultWorkflowTemplates } from './workflow-manager';  
export type { WorkflowTemplate, WorkflowLibrary, Workflow } from './workflow-manager'; 
