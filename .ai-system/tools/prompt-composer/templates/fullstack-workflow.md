# Fullstack Development Workflow

**Project:** {{project_name}}
**Stack:** {{tech_stack}}

---

## Phase 1: Architecture

{{include:superprompts/monorepo-architecture.md}}

---

## Phase 2: Backend Development

{{component:code-review-header|language={{backend_lang}}}}

**Requirements:**
- RESTful API design
- Database schema
- Authentication

{{component:testing-requirements|coverage=85}}

---

## Phase 3: Frontend Development

{{component:code-review-header|language={{frontend_lang}}}}

**Requirements:**
- Component architecture
- State management
- Responsive design

---

## Phase 4: Testing & Deployment

{{component:security-checklist}}

{{include:superprompts/cicd-pipeline-setup.md}}

---

## Success Criteria

- All tests passing
- {{coverage_target}}% code coverage
- Performance benchmarks met
- Security audit passed
