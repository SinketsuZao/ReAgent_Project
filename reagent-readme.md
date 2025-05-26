# ReAgent: Reversible Multi-Agent Reasoning System

<div align="center">

![ReAgent Logo](docs/images/reagent-logo.png)

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Documentation](https://img.shields.io/badge/docs-available-orange.svg)](docs/)

*A production-ready implementation of reversible multi-agent reasoning for knowledge-enhanced multi-hop question answering*

</div>

## 🌟 Overview

ReAgent is an advanced multi-agent system that implements reversible reasoning mechanisms for complex question-answering tasks. Based on cutting-edge research in multi-hop QA, ReAgent introduces explicit backtracking protocols that allow agents to correct errors during inference, leading to more accurate and reliable answers.

### Key Features

- 🔄 **Reversible Reasoning**: Local and global backtracking mechanisms to correct errors
- 🤖 **Multi-Agent Architecture**: Six specialized agents working collaboratively
- 📊 **Production Ready**: Complete with monitoring, logging, and deployment configurations
- 🚀 **Scalable**: Distributed processing with Celery and Redis
- 📈 **Observable**: Prometheus metrics and Grafana dashboards
- 🔐 **Secure**: Environment-based configuration and authentication support

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Interaction Layer              │
│  ┌─────────┐ ┌─────────┐ ┌──────────────┐ │
│  │  Logs   │ │ Tracker │ │   Messages   │ │
│  └─────────┘ └─────────┘ └──────────────┘ │
├─────────────────────────────────────────────┤
│             Supervisor Layer                │
│  ┌─────────────┐         ┌──────────────┐ │
│  │ Supervisor  │         │  Controller  │ │
│  └─────────────┘         └──────────────┘ │
├─────────────────────────────────────────────┤
│              Execution Layer                │
│  ┌────┐  ┌────┐  ┌────┐  ┌─────────────┐ │
│  │ Q  │  │ R  │  │ V  │  │ Assembler   │ │
│  └────┘  └────┘  └────┘  └─────────────┘ │
└─────────────────────────────────────────────┘
```

### Agent Roles

1. **Question Decomposer (Q)**: Breaks complex questions into manageable sub-questions
2. **Retriever (R)**: Fetches relevant evidence from knowledge sources
3. **Verifier (V)**: Validates consistency and triggers local backtracking
4. **Answer Assembler (A)**: Synthesizes partial answers into final response
5. **Supervisor (S)**: Manages global conflicts and system-wide rollback
6. **Controller (C)**: Provides strategic oversight and intervention

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- OpenAI API key
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/reagent.git
   cd reagent
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Start the system**
   ```bash
   docker-compose up -d
   ```

4. **Verify installation**
   ```bash
   curl http://localhost:8000/health
   ```

### Your First Query

```bash
curl -X POST http://localhost:8000/api/v1/questions \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which U.S. state has a capital city whose population is smaller than the states largest city, given that this state hosted the 1984 Summer Olympics?"
  }'
```

## 📖 Documentation

- [Architecture Guide](docs/architecture.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)
- [Development Guide](docs/development.md)

## 🛠️ Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run linting
black reagent/ api/ tests/
flake8 reagent/ api/
mypy reagent/ api/
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Coverage report
pytest --cov=reagent --cov-report=html
```

### Code Style

We use:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking
- `pre-commit` for git hooks

## 📊 Monitoring

### Metrics

Access Prometheus metrics at `http://localhost:9090`

Key metrics:
- `reagent_questions_processed_total`
- `reagent_backtracking_total`
- `reagent_agent_processing_seconds`
- `reagent_llm_tokens_used`

### Dashboards

Access Grafana at `http://localhost:3000` (default: admin/admin)

Available dashboards:
- System Overview
- Agent Performance
- Backtracking Analysis
- Token Usage

## 🚢 Deployment

### Docker Deployment

```bash
# Production build
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale workers
docker-compose up -d --scale reagent_worker=5
```

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/

# Check status
kubectl get pods -n reagent
```

### Environment Variables

Key configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `LLM_MODEL` | Model to use | gpt-4o |
| `MAX_BACKTRACK_DEPTH` | Max backtracking depth | 5 |
| `POSTGRES_PASSWORD` | Database password | Required |

See `.env.example` for full list.

## 🔧 Troubleshooting

### Common Issues

1. **LLM API Errors**
   - Check API key in `.env`
   - Verify rate limits
   - Check network connectivity

2. **High Latency**
   - Reduce `MAX_BACKTRACK_DEPTH`
   - Check Redis performance
   - Monitor token usage

3. **Database Connection Issues**
   ```bash
   docker-compose logs postgres
   docker-compose exec postgres psql -U reagent_user -d reagent_db
   ```

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG docker-compose up
```

## 📈 Performance

### Benchmarks

| Dataset | EM Score | F1 Score | Avg. Time |
|---------|----------|----------|-----------|
| HotpotQA | 0.630 | 0.795 | 12.3s |
| 2Wiki | 0.711 | 0.793 | 15.7s |
| MuSiQue | 0.371 | 0.515 | 18.2s |

### Optimization Tips

1. **Caching**: Enable Redis caching for repeated queries
2. **Batching**: Process multiple sub-questions in parallel
3. **Model Selection**: Use lighter models for simple tasks
4. **Resource Limits**: Set appropriate token budgets

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use ReAgent in your research, please cite:

```bibtex
@article{reagent2024,
  title={ReAgent: Reversible Multi-Agent Reasoning for Knowledge-Enhanced Multi-Hop QA},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 🙏 Acknowledgments

- OpenAI for GPT models
- The open-source community
- Research papers that inspired this work

## 📞 Support

- 📧 Email: support@reagent.ai
- 💬 Discord: [Join our server](https://discord.gg/reagent)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/reagent/issues)
- 📖 Docs: [Full Documentation](https://docs.reagent.ai)

---

<div align="center">
Made with ❤️ by the ReAgent Team
</div>