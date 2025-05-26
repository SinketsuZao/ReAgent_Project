#!/bin/bash

# ReAgent System Setup Script
# This script sets up the development environment for the ReAgent system

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="3.11"
VENV_DIR="venv"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

# Functions
print_header() {
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed"
        return 1
    else
        print_success "$1 is installed"
        return 0
    fi
}

check_python_version() {
    if command -v python3 &> /dev/null; then
        PYTHON_CURRENT=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [ "$(printf '%s\n' "$PYTHON_VERSION" "$PYTHON_CURRENT" | sort -V | head -n1)" = "$PYTHON_VERSION" ]; then
            print_success "Python $PYTHON_CURRENT (>= $PYTHON_VERSION required)"
            return 0
        else
            print_error "Python $PYTHON_CURRENT found, but $PYTHON_VERSION+ required"
            return 1
        fi
    else
        print_error "Python 3 is not installed"
        return 1
    fi
}

create_directories() {
    print_header "Creating project directories"
    
    directories=(
        "logs"
        "data"
        "configs"
        "db/migrations/versions"
        "monitoring/grafana/dashboards"
        "monitoring/grafana/provisioning"
        "monitoring/alerting"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$PROJECT_ROOT/$dir" ]; then
            mkdir -p "$PROJECT_ROOT/$dir"
            print_success "Created $dir"
        else
            print_warning "$dir already exists"
        fi
    done
}

setup_environment_file() {
    print_header "Setting up environment configuration"
    
    cd "$PROJECT_ROOT"
    
    if [ -f "$ENV_FILE" ]; then
        print_warning "$ENV_FILE already exists"
        read -p "Do you want to backup and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
            print_success "Backed up existing $ENV_FILE"
        else
            return 0
        fi
    fi
    
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        print_success "Created $ENV_FILE from $ENV_EXAMPLE"
        
        # Prompt for critical values
        echo
        print_warning "Please configure the following required settings:"
        echo
        
        # OpenAI API Key
        read -p "Enter your OpenAI API key (or press Enter to skip): " OPENAI_KEY
        if [ -n "$OPENAI_KEY" ]; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_KEY/" "$ENV_FILE"
            else
                sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_KEY/" "$ENV_FILE"
            fi
            print_success "Set OpenAI API key"
        fi
        
        # Database password
        read -p "Enter PostgreSQL password (or press Enter for default): " DB_PASS
        if [ -n "$DB_PASS" ]; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$DB_PASS/" "$ENV_FILE"
            else
                sed -i "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$DB_PASS/" "$ENV_FILE"
            fi
            print_success "Set PostgreSQL password"
        fi
        
        # Environment
        read -p "Enter environment (development/production) [development]: " ENV_TYPE
        ENV_TYPE=${ENV_TYPE:-development}
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/ENVIRONMENT=.*/ENVIRONMENT=$ENV_TYPE/" "$ENV_FILE"
        else
            sed -i "s/ENVIRONMENT=.*/ENVIRONMENT=$ENV_TYPE/" "$ENV_FILE"
        fi
        print_success "Set environment to $ENV_TYPE"
        
    else
        print_error "$ENV_EXAMPLE not found"
        return 1
    fi
}

setup_python_environment() {
    print_header "Setting up Python environment"
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        print_success "Created virtual environment"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Activated virtual environment"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    print_success "Upgraded pip, setuptools, and wheel"
    
    # Install dependencies
    if [ -f "requirements.txt" ]; then
        print_warning "Installing dependencies (this may take a while)..."
        pip install -r requirements.txt
        print_success "Installed Python dependencies"
    else
        print_error "requirements.txt not found"
        return 1
    fi
    
    # Install development dependencies
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        print_success "Installed development dependencies"
    fi
    
    # Install package in editable mode
    if [ -f "setup.py" ]; then
        pip install -e .
        print_success "Installed ReAgent package in editable mode"
    fi
}

setup_docker_services() {
    print_header "Setting up Docker services"
    
    cd "$PROJECT_ROOT"
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker and try again."
        return 1
    fi
    
    # Pull required images
    print_warning "Pulling Docker images..."
    docker-compose pull
    print_success "Pulled Docker images"
    
    # Start infrastructure services
    print_warning "Starting infrastructure services..."
    docker-compose up -d redis postgres elasticsearch
    
    # Wait for services to be ready
    print_warning "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    if docker-compose ps | grep -q "Up"; then
        print_success "Docker services are running"
    else
        print_error "Some Docker services failed to start"
        docker-compose ps
        return 1
    fi
}

setup_database() {
    print_header "Setting up database"
    
    cd "$PROJECT_ROOT"
    
    # Wait for PostgreSQL to be ready
    print_warning "Waiting for PostgreSQL to be ready..."
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U reagent_user > /dev/null 2>&1; then
            print_success "PostgreSQL is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "PostgreSQL failed to start"
            return 1
        fi
        sleep 2
    done
    
    # Run database initialization
    if [ -f "db/init.sql" ]; then
        print_warning "Initializing database schema..."
        docker-compose exec -T postgres psql -U reagent_user -d reagent_db < db/init.sql
        print_success "Database schema initialized"
    else
        print_warning "db/init.sql not found, skipping database initialization"
    fi
    
    # Run Alembic migrations
    if [ -f "alembic.ini" ]; then
        print_warning "Running database migrations..."
        source "$VENV_DIR/bin/activate"
        alembic upgrade head
        print_success "Database migrations completed"
    fi
}

setup_monitoring() {
    print_header "Setting up monitoring"
    
    cd "$PROJECT_ROOT"
    
    # Create Prometheus data directory
    mkdir -p monitoring/prometheus/data
    chmod 777 monitoring/prometheus/data
    
    # Create Grafana data directory
    mkdir -p monitoring/grafana/data
    chmod 777 monitoring/grafana/data
    
    # Start monitoring services
    print_warning "Starting monitoring services..."
    docker-compose up -d prometheus grafana
    
    sleep 5
    
    if docker-compose ps | grep -E "prometheus.*Up" | grep -q "Up"; then
        print_success "Prometheus is running at http://localhost:9090"
    fi
    
    if docker-compose ps | grep -E "grafana.*Up" | grep -q "Up"; then
        print_success "Grafana is running at http://localhost:3000 (admin/admin)"
    fi
}

run_tests() {
    print_header "Running tests"
    
    cd "$PROJECT_ROOT"
    source "$VENV_DIR/bin/activate"
    
    # Run unit tests
    print_warning "Running unit tests..."
    if python -m pytest tests/unit -v --tb=short; then
        print_success "Unit tests passed"
    else
        print_warning "Some unit tests failed"
    fi
    
    # Check if integration tests should run
    read -p "Run integration tests? (requires all services) (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Running integration tests..."
        if python -m pytest tests/integration -v --tb=short; then
            print_success "Integration tests passed"
        else
            print_warning "Some integration tests failed"
        fi
    fi
}

generate_ssl_certificates() {
    print_header "Generating SSL certificates (development)"
    
    cd "$PROJECT_ROOT"
    
    SSL_DIR="configs/ssl"
    mkdir -p "$SSL_DIR"
    
    if [ ! -f "$SSL_DIR/privkey.pem" ]; then
        print_warning "Generating self-signed SSL certificate..."
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$SSL_DIR/privkey.pem" \
            -out "$SSL_DIR/fullchain.pem" \
            -subj "/C=US/ST=State/L=City/O=ReAgent/CN=localhost"
        
        cp "$SSL_DIR/fullchain.pem" "$SSL_DIR/chain.pem"
        print_success "Generated SSL certificates"
    else
        print_warning "SSL certificates already exist"
    fi
}

print_next_steps() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}The ReAgent system has been set up successfully!${NC}"
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   ${BLUE}source $VENV_DIR/bin/activate${NC}"
    echo
    echo "2. Start all services:"
    echo "   ${BLUE}docker-compose up -d${NC}"
    echo
    echo "3. Run the API server:"
    echo "   ${BLUE}uvicorn api.main:app --reload${NC}"
    echo
    echo "4. Run a Celery worker:"
    echo "   ${BLUE}celery -A worker.celery_app worker --loglevel=info${NC}"
    echo
    echo "5. Access the services:"
    echo "   - API: http://localhost:8000"
    echo "   - API Docs: http://localhost:8000/docs"
    echo "   - Prometheus: http://localhost:9090"
    echo "   - Grafana: http://localhost:3000 (admin/admin)"
    echo
    echo "6. Process a question:"
    echo "   ${BLUE}curl -X POST http://localhost:8000/api/v1/questions \\
     -H \"Content-Type: application/json\" \\
     -d '{\"question\": \"What is the capital of France?\"}'${NC}"
    echo
    echo "For more information, see the README.md file."
}

# Main execution
main() {
    print_header "ReAgent System Setup"
    
    echo "This script will set up the ReAgent development environment."
    echo "It will install dependencies, create directories, and start services."
    echo
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    
    # Check prerequisites
    print_header "Checking prerequisites"
    
    PREREQ_MET=true
    
    check_command "git" || PREREQ_MET=false
    check_command "docker" || PREREQ_MET=false
    check_command "docker-compose" || PREREQ_MET=false
    check_python_version || PREREQ_MET=false
    check_command "psql" || print_warning "psql not found (optional)"
    check_command "redis-cli" || print_warning "redis-cli not found (optional)"
    
    if [ "$PREREQ_MET" = false ]; then
        print_error "Please install missing prerequisites and try again."
        exit 1
    fi
    
    # Run setup steps
    create_directories
    setup_environment_file
    setup_python_environment
    generate_ssl_certificates
    setup_docker_services
    setup_database
    setup_monitoring
    run_tests
    
    # Print completion message
    print_next_steps
}

# Run main function
main "$@"
