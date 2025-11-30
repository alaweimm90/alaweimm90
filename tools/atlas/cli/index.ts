// ATLAS CLI Entry Point  
  
import { createCLI } from './commands.js';  
  
// Create and run the CLI  
const program = createCLI();  
  
// Parse command line arguments  
program.parse(); 
