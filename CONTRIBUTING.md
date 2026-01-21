# Contributing Guidelines

Thank you for taking on this challenge! This document explains how to submit your solution.

## 🌿 Branch Strategy

### For Candidates

1. **Fork the repository** to your GitHub account

2. **Clone your fork:**
   ```bash
   git clone https://github.com/gabo-di/RoboAI.git
   cd RoboAI
   ```

3. **Create a solution branch:**
   ```bash
   git checkout -b solution/firstname-lastname
   # Example: solution/jane-doe
   ```

4. **Work on your solution** with regular commits

5. **Push and create a Pull Request** against the `main` branch of the original repo

### Branch Naming Convention

| Branch Type | Pattern | Example |
|-------------|---------|---------|
| Your solution | `solution/firstname-lastname` | `solution/jane-doe` |
| Feature addition | `feat/description` | `feat/add-docker-support` |
| Bug fix | `fix/description` | `fix/angle-wrapping` |
| Documentation | `docs/description` | `docs/add-solution-report` |
| Tests | `test/description` | `test/add-simulator-tests` |
| Refactor | `refactor/description` | `refactor/config-extraction` |

## 📝 Commit Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature or enhancement |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `test` | Adding or updating tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `style` | Formatting, missing semicolons, etc. |
| `perf` | Performance improvements |
| `chore` | Maintenance tasks |

### Examples

```bash
# Good commit messages
git commit -m "fix: correct motor imbalance in realistic simulation"
git commit -m "feat: add sin/cos angle preprocessing for better learning"
git commit -m "docs: document discovered issues and solutions"
git commit -m "test: add unit tests for trajectory splitting"
git commit -m "refactor: extract hyperparameters to config file"

# Bad commit messages
git commit -m "fixed stuff"
git commit -m "WIP"
git commit -m "changes"
```

## 📋 Pull Request Process

### Before Submitting

- [ ] Code runs without errors
- [ ] All existing tests pass (if any)
- [ ] New functionality has tests (bonus points!)
- [ ] Code is formatted consistently
- [ ] Documentation is updated
- [ ] Solution report is complete (`docs/SOLUTION_REPORT.md`)

### PR Title Format

```
[Solution] Your Name - Brief Description
```

Example: `[Solution] Jane Doe - Fixed sim2real gap and data leakage`

### PR Description Template

Your PR description should include:

```markdown
## Summary
Brief overview of your solution approach.

## Issues Discovered
1. Issue 1: Description
2. Issue 2: Description

## Solutions Implemented
1. Fix for Issue 1
2. Fix for Issue 2

## Results
- Before: X% success rate
- After: Y% success rate

## Side Quests Completed (if any)
- [ ] Docker container
- [ ] Unit tests
- [ ] etc.

## Time Spent
Approximately X hours
```

## 💻 Development Setup

### Code Style

We recommend using these tools (not required, but appreciated):

```bash
# Install development dependencies
pip install black isort flake8 mypy

# Format code
black src/
isort src/

# Check style
flake8 src/
mypy src/
```

### Pre-commit Hooks (Optional Bonus)

```bash
pip install pre-commit
pre-commit install
```

## 📊 What Makes a Great Submission

### Must Have
- Clear documentation of findings
- Working code
- Explanation of debugging process

### Nice to Have
- Clean commit history
- Well-structured code changes
- Unit tests for fixes

### Impressive
- Multiple valid solution approaches compared
- Performance analysis
- Side quests completed
- Creative solutions we didn't expect!

## ❓ Questions?

- **Technical questions:** Open an issue with `question` label
- **Challenge clarifications:** Open an issue with `clarification` label

---

Good luck with your submission! 🚀
