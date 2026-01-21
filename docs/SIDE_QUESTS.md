# 🎮 Side Quests (Bonus Challenges)

These optional challenges demonstrate additional skills valued in robotics engineering. Complete any that interest you for bonus points!

**Note:** Focus on the main challenge first. Side quests are meant to showcase additional skills, not replace the core task.

---

## 🐳 Quest 1: Docker Container
**Difficulty:** ⭐⭐☆☆☆
**Skills:** DevOps, Reproducibility

Create a Docker container that runs the entire pipeline.

### Requirements
- [ ] `Dockerfile` that builds the environment
- [ ] `docker-compose.yml` for easy execution
- [ ] Container can run training and evaluation
- [ ] Results are accessible from host machine (volume mounts)

### Deliverables
```
docker/
├── Dockerfile
├── docker-compose.yml
└── README.md          # Instructions to build and run
```

### Bonus Points
- Multi-stage build for smaller image
- GPU support (nvidia-docker)
- Health checks

---
