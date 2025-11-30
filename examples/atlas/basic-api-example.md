# Basic API Example

Complete example of using ATLAS to build a REST API for a task management application, demonstrating code generation, review, testing, and refactoring capabilities.

---

## Overview

This example demonstrates how to use ATLAS to build a complete REST API for a task management application. We'll create:

- User authentication endpoints
- Task CRUD operations
- Input validation
- Error handling
- Database integration
- API documentation

---

## Prerequisites

```bash
# Install ATLAS CLI
npm install -g @atlas/cli

# Initialize project
mkdir task-api && cd task-api
npm init -y
npm install express mongoose bcryptjs jsonwebtoken dotenv cors helmet

# Initialize ATLAS
atlas init
```

---

## Step 1: Register AI Agents

Register multiple agents for different types of work:

```bash
# Primary agent for code generation (Claude)
atlas agent register claude-sonnet-4 \
  --name "Claude Sonnet 4" \
  --provider anthropic \
  --capabilities code_generation,code_review,refactoring,debugging \
  --api-key YOUR_ANTHROPIC_KEY

# Secondary agent for reviews (GPT-4)
atlas agent register gpt-4-turbo \
  --name "GPT-4 Turbo" \
  --provider openai \
  --capabilities code_review,security_analysis,architecture \
  --api-key YOUR_OPENAI_KEY

# Tertiary agent for testing (Gemini)
atlas agent register gemini-pro \
  --name "Gemini Pro" \
  --provider google \
  --capabilities testing,documentation \
  --api-key YOUR_GOOGLE_KEY
```

---

## Step 2: Create Project Structure

Generate the basic project structure:

```bash
atlas task submit \
  --type code_generation \
  --description "Create basic Express.js project structure with MVC pattern for task management API" \
  --context language=javascript,framework=express,database=mongodb,pattern=mvc \
  --priority high
```

**Expected Output:**
```
📁 project structure created:
├── src/
│   ├── controllers/
│   ├── models/
│   ├── routes/
│   ├── middleware/
│   ├── config/
│   └── utils/
├── tests/
├── docs/
└── server.js
```

---

## Step 3: Database Models

Generate the database models:

```bash
atlas task submit \
  --type code_generation \
  --description "Create Mongoose models for User and Task with validation and relationships" \
  --context database=mongodb,odm=mongoose,features=validation,indexes,timestamps \
  --file-path src/models/
```

**Generated User Model:**
```javascript
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    minlength: 3,
    maxlength: 50
  },
  email: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    lowercase: true,
    match: [/^\w+([.-]?\w+)*@\w+([.-]?\w+)*(\.\w{2,3})+$/, 'Please enter a valid email']
  },
  password: {
    type: String,
    required: true,
    minlength: 6
  },
  role: {
    type: String,
    enum: ['user', 'admin'],
    default: 'user'
  }
}, {
  timestamps: true
});

// Hash password before saving
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});

// Compare password method
userSchema.methods.comparePassword = async function(candidatePassword) {
  return bcrypt.compare(candidatePassword, this.password);
};

module.exports = mongoose.model('User', userSchema);
```

**Generated Task Model:**
```javascript
const mongoose = require('mongoose');

const taskSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
    trim: true,
    maxlength: 200
  },
  description: {
    type: String,
    trim: true,
    maxlength: 2000
  },
  status: {
    type: String,
    enum: ['pending', 'in_progress', 'completed', 'cancelled'],
    default: 'pending'
  },
  priority: {
    type: String,
    enum: ['low', 'medium', 'high', 'urgent'],
    default: 'medium'
  },
  assignedTo: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  createdBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  dueDate: {
    type: Date
  },
  tags: [{
    type: String,
    trim: true,
    lowercase: true
  }],
  completedAt: {
    type: Date
  }
}, {
  timestamps: true
});

// Index for performance
taskSchema.index({ assignedTo: 1, status: 1 });
taskSchema.index({ createdBy: 1, createdAt: -1 });
taskSchema.index({ tags: 1 });

module.exports = mongoose.model('Task', taskSchema);
```

---

## Step 4: Authentication Middleware

Generate JWT authentication middleware:

```bash
atlas task submit \
  --type code_generation \
  --description "Create JWT authentication middleware with role-based access control" \
  --context auth=jwt,roles=user,admin,security=helmet,cors \
  --file-path src/middleware/auth.js
```

**Generated Authentication Middleware:**
```javascript
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const auth = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');

    if (!token) {
      return res.status(401).json({ error: 'Access denied. No token provided.' });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user = await User.findById(decoded.userId);

    if (!user) {
      return res.status(401).json({ error: 'Invalid token. User not found.' });
    }

    req.user = user;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token.' });
  }
};

const requireRole = (roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required.' });
    }

    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Insufficient permissions.' });
    }

    next();
  };
};

const optionalAuth = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');

    if (token) {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      const user = await User.findById(decoded.userId);
      if (user) {
        req.user = user;
      }
    }

    next();
  } catch (error) {
    next();
  }
};

module.exports = { auth, requireRole, optionalAuth };
```

---

## Step 5: Authentication Routes

Generate authentication endpoints:

```bash
atlas task submit \
  --type code_generation \
  --description "Create authentication routes for user registration, login, and profile management" \
  --context auth=jwt,validation=joi,password=bcrypt,security=rate-limiting \
  --file-path src/routes/auth.js
```

**Generated Auth Routes:**
```javascript
const express = require('express');
const Joi = require('joi');
const User = require('../models/User');
const { auth } = require('../middleware/auth');

const router = express.Router();

// Validation schemas
const registerSchema = Joi.object({
  username: Joi.string().min(3).max(50).required(),
  email: Joi.string().email().required(),
  password: Joi.string().min(6).required()
});

const loginSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().required()
});

// Register new user
router.post('/register', async (req, res) => {
  try {
    const { error } = registerSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { username, email, password } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({
      $or: [{ email }, { username }]
    });

    if (existingUser) {
      return res.status(409).json({ error: 'User already exists' });
    }

    const user = new User({ username, email, password });
    await user.save();

    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    );

    res.status(201).json({
      message: 'User registered successfully',
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        role: user.role
      },
      token
    });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Login user
router.post('/login', async (req, res) => {
  try {
    const { error } = loginSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const isValidPassword = await user.comparePassword(password);
    if (!isValidPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    );

    res.json({
      message: 'Login successful',
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        role: user.role
      },
      token
    });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Get current user profile
router.get('/profile', auth, async (req, res) => {
  try {
    const user = await User.findById(req.user._id).select('-password');
    res.json({ user });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Update user profile
router.put('/profile', auth, async (req, res) => {
  try {
    const allowedUpdates = ['username', 'email'];
    const updates = {};

    for (const field of allowedUpdates) {
      if (req.body[field] !== undefined) {
        updates[field] = req.body[field];
      }
    }

    const user = await User.findByIdAndUpdate(
      req.user._id,
      updates,
      { new: true, runValidators: true }
    ).select('-password');

    res.json({
      message: 'Profile updated successfully',
      user
    });
  } catch (error) {
    if (error.code === 11000) {
      res.status(409).json({ error: 'Username or email already exists' });
    } else {
      res.status(500).json({ error: 'Server error' });
    }
  }
});

module.exports = router;
```

---

## Step 6: Task Management Routes

Generate task CRUD operations:

```bash
atlas task submit \
  --type code_generation \
  --description "Create complete CRUD routes for task management with proper validation and authorization" \
  --context operations=create,read,update,delete,filtering,pagination,sorting \
  --file-path src/routes/tasks.js
```

**Generated Task Routes:**
```javascript
const express = require('express');
const Joi = require('joi');
const Task = require('../models/Task');
const { auth, requireRole } = require('../middleware/auth');

const router = express.Router();

// Validation schemas
const createTaskSchema = Joi.object({
  title: Joi.string().min(1).max(200).required(),
  description: Joi.string().max(2000).allow(''),
  priority: Joi.string().valid('low', 'medium', 'high', 'urgent').default('medium'),
  assignedTo: Joi.string().required(),
  dueDate: Joi.date().iso().allow(null),
  tags: Joi.array().items(Joi.string()).default([])
});

const updateTaskSchema = Joi.object({
  title: Joi.string().min(1).max(200),
  description: Joi.string().max(2000).allow(''),
  status: Joi.string().valid('pending', 'in_progress', 'completed', 'cancelled'),
  priority: Joi.string().valid('low', 'medium', 'high', 'urgent'),
  assignedTo: Joi.string(),
  dueDate: Joi.date().iso().allow(null),
  tags: Joi.array().items(Joi.string())
});

// Get all tasks with filtering and pagination
router.get('/', auth, async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const skip = (page - 1) * limit;

    // Build filter
    const filter = {};
    if (req.query.status) filter.status = req.query.status;
    if (req.query.priority) filter.priority = req.query.priority;
    if (req.query.assignedTo) filter.assignedTo = req.query.assignedTo;
    if (req.query.createdBy) filter.createdBy = req.query.createdBy;
    if (req.query.tags) {
      filter.tags = { $in: req.query.tags.split(',') };
    }

    // Only show tasks user can access
    if (req.user.role !== 'admin') {
      filter.$or = [
        { assignedTo: req.user._id },
        { createdBy: req.user._id }
      ];
    }

    // Build sort
    const sort = {};
    const sortBy = req.query.sortBy || 'createdAt';
    const sortOrder = req.query.sortOrder === 'asc' ? 1 : -1;
    sort[sortBy] = sortOrder;

    const tasks = await Task.find(filter)
      .populate('assignedTo', 'username email')
      .populate('createdBy', 'username email')
      .sort(sort)
      .skip(skip)
      .limit(limit);

    const total = await Task.countDocuments(filter);

    res.json({
      tasks,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Get single task
router.get('/:id', auth, async (req, res) => {
  try {
    const task = await Task.findById(req.params.id)
      .populate('assignedTo', 'username email')
      .populate('createdBy', 'username email');

    if (!task) {
      return res.status(404).json({ error: 'Task not found' });
    }

    // Check permissions
    if (req.user.role !== 'admin' &&
        task.assignedTo._id.toString() !== req.user._id.toString() &&
        task.createdBy._id.toString() !== req.user._id.toString()) {
      return res.status(403).json({ error: 'Access denied' });
    }

    res.json({ task });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Create new task
router.post('/', auth, async (req, res) => {
  try {
    const { error } = createTaskSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const taskData = {
      ...req.body,
      createdBy: req.user._id
    };

    const task = new Task(taskData);
    await task.save();

    await task.populate('assignedTo', 'username email');
    await task.populate('createdBy', 'username email');

    res.status(201).json({
      message: 'Task created successfully',
      task
    });
  } catch (error) {
    if (error.name === 'ValidationError') {
      res.status(400).json({ error: error.message });
    } else {
      res.status(500).json({ error: 'Server error' });
    }
  }
});

// Update task
router.put('/:id', auth, async (req, res) => {
  try {
    const { error } = updateTaskSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const task = await Task.findById(req.params.id);
    if (!task) {
      return res.status(404).json({ error: 'Task not found' });
    }

    // Check permissions
    if (req.user.role !== 'admin' &&
        task.createdBy.toString() !== req.user._id.toString()) {
      return res.status(403).json({ error: 'Access denied' });
    }

    // Handle status change to completed
    if (req.body.status === 'completed' && task.status !== 'completed') {
      req.body.completedAt = new Date();
    } else if (req.body.status !== 'completed') {
      req.body.completedAt = undefined;
    }

    Object.assign(task, req.body);
    await task.save();

    await task.populate('assignedTo', 'username email');
    await task.populate('createdBy', 'username email');

    res.json({
      message: 'Task updated successfully',
      task
    });
  } catch (error) {
    if (error.name === 'ValidationError') {
      res.status(400).json({ error: error.message });
    } else {
      res.status(500).json({ error: 'Server error' });
    }
  }
});

// Delete task
router.delete('/:id', auth, async (req, res) => {
  try {
    const task = await Task.findById(req.params.id);
    if (!task) {
      return res.status(404).json({ error: 'Task not found' });
    }

    // Check permissions
    if (req.user.role !== 'admin' &&
        task.createdBy.toString() !== req.user._id.toString()) {
      return res.status(403).json({ error: 'Access denied' });
    }

    await Task.findByIdAndDelete(req.params.id);

    res.json({ message: 'Task deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

module.exports = router;
```

---

## Step 7: Main Server File

Generate the main server application:

```bash
atlas task submit \
  --type code_generation \
  --description "Create main Express server with middleware setup, route configuration, and error handling" \
  --context server=express,middleware=helmet,cors,logging,database=mongo \
  --file-path server.js
```

---

## Step 8: Code Review

Review the generated code for quality and security:

```bash
atlas task submit \
  --type code_review \
  --description "Comprehensive security and code quality review of the generated API" \
  --files src/routes/auth.js,src/routes/tasks.js,src/models/User.js,src/models/Task.js \
  --focus security,performance,best-practices
```

---

## Step 9: Generate Tests

Create comprehensive tests:

```bash
atlas task submit \
  --type testing \
  --description "Generate comprehensive unit and integration tests for authentication and task management" \
  --context framework=jest,supertest,mongo-memory-server \
  --files src/routes/auth.js,src/routes/tasks.js
```

---

## Step 10: API Documentation

Generate API documentation:

```bash
atlas task submit \
  --type documentation \
  --description "Create comprehensive API documentation with examples and schemas" \
  --context format=openapi,schema=json \
  --files src/routes/auth.js,src/routes/tasks.js
```

---

## Step 11: Repository Analysis

Analyze the codebase for improvements:

```bash
atlas analyze repo . --type full --output analysis.json
```

---

## Step 12: Apply Refactorings

Apply suggested improvements:

```bash
atlas refactor list --analysis-id <analysis-id>
atlas refactor apply <opportunity-id> --create-pr
```

---

## Final Project Structure

```
task-api/
├── src/
│   ├── controllers/
│   ├── models/
│   │   ├── User.js
│   │   └── Task.js
│   ├── routes/
│   │   ├── auth.js
│   │   └── tasks.js
│   ├── middleware/
│   │   └── auth.js
│   ├── config/
│   │   └── database.js
│   └── utils/
├── tests/
│   ├── auth.test.js
│   ├── tasks.test.js
│   └── integration.test.js
├── docs/
│   └── api.yaml
├── .atlas/
│   └── config.json
├── server.js
├── package.json
└── README.md
```

---

## Running the API

```bash
# Install dependencies
npm install

# Set environment variables
cp .env.example .env
# Edit .env with your MongoDB URL and JWT secret

# Start the server
npm start

# Run tests
npm test

# Run with ATLAS monitoring
atlas optimize start --repository . --schedule hourly
```

---

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `GET /api/auth/profile` - Get user profile
- `PUT /api/auth/profile` - Update user profile

### Tasks
- `GET /api/tasks` - List tasks (with filtering/pagination)
- `GET /api/tasks/:id` - Get single task
- `POST /api/tasks` - Create new task
- `PUT /api/tasks/:id` - Update task
- `DELETE /api/tasks/:id` - Delete task

---

## Next Steps

1. **Add More Features**: File uploads, notifications, task comments
2. **Deploy**: Set up CI/CD pipeline with ATLAS integration
3. **Scale**: Add caching, rate limiting, database optimization
4. **Monitor**: Set up comprehensive monitoring and alerting
5. **Security**: Add API rate limiting, input sanitization, audit logs

This example demonstrates how ATLAS can accelerate development from concept to production-ready API, with built-in code review, testing, and optimization capabilities.</instructions>