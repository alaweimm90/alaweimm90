// Task Template
module.exports = {
  name: 'task-name',
  description: 'Task description',
  schedule: '0 0 * * *', // Daily at midnight

  async execute(context) {
    // Task implementation
    return { success: true };
  }
};