#!/bin/bash

# ReAgent System Deployment Script
# This script handles deployment to various environments

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEPLOYMENT_LOG="$PROJECT_ROOT/logs/deployment_${TIMESTAMP}.log"

# Default values
ENVIRONMENT=""
DEPLOYMENT_TYPE=""
VERSION=""
DOCKER_REGISTRY=""
K8S_NAMESPACE="reagent"
ROLLBACK_VERSION=""

# Functions
print_header() {
    echo -e "${BLUE}===============================================${NC}" | tee -a "$DEPLOYMENT_LOG"
    echo -e "${BLUE}$1${NC}" | tee -a "$DEPLOYMENT_LOG"
    echo -e "${BLUE}===============================================${NC}" | tee -a "$DEPLOYMENT_LOG"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$DEPLOYMENT_LOG"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "$DEPLOYMENT_LOG"
}

print_error() {
    echo -e "${RED}✗ $1${NC}" | tee -a "$DEPLOYMENT_LOG"
}

print_info() {
    echo -e "${PURPLE}ℹ $1${NC}" | tee -a "$DEPLOYMENT_LOG"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -e, --environment ENV     Target environment (development/staging/production)"
    echo "  -t, --type TYPE          Deployment type (docker/kubernetes/docker-swarm)"
    echo "  -v, --version VERSION    Version to deploy (e.g., v1.0.0)"
    echo "  -r, --registry REGISTRY  Docker registry URL"
    echo "  --rollback VERSION       Rollback to specified version"
    echo "  -h, --help              Show this help message"
    echo
    echo "Examples:"
    echo "  $0 -e production -t kubernetes -v v1.0.0 -r docker.io/reagent"
    echo "  $0 -e staging -t docker -v latest"
    echo "  $0 -e production -t kubernetes --rollback v0.9.5"
    exit 1
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            --rollback)
                ROLLBACK_VERSION="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Validate required arguments
    if [ -z "$ENVIRONMENT" ]; then
        print_error "Environment is required"
        usage
    fi
    
    if [ -z "$DEPLOYMENT_TYPE" ]; then
        print_error "Deployment type is required"
        usage
    fi
    
    if [ -z "$VERSION" ] && [ -z "$ROLLBACK_VERSION" ]; then
        print_error "Version or rollback version is required"
        usage
    fi
}

validate_environment() {
    print_header "Validating Environment"
    
    case $ENVIRONMENT in
        development|staging|production)
            print_success "Environment: $ENVIRONMENT"
            ;;
        *)
            print_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    case $DEPLOYMENT_TYPE in
        docker|kubernetes|docker-swarm)
            print_success "Deployment type: $DEPLOYMENT_TYPE"
            ;;
        *)
            print_error "Invalid deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    # Check required tools
    case $DEPLOYMENT_TYPE in
        docker|docker-swarm)
            if ! command -v docker &> /dev/null; then
                print_error "Docker is not installed"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                print_error "kubectl is not installed"
                exit 1
            fi
            ;;
    esac
}

load_environment_config() {
    print_header "Loading Environment Configuration"
    
    ENV_FILE="$PROJECT_ROOT/.env.$ENVIRONMENT"
    if [ ! -f "$ENV_FILE" ]; then
        print_warning "Environment file $ENV_FILE not found, using .env"
        ENV_FILE="$PROJECT_ROOT/.env"
    fi
    
    if [ -f "$ENV_FILE" ]; then
        set -a
        source "$ENV_FILE"
        set +a
        print_success "Loaded environment configuration from $ENV_FILE"
    else
        print_error "No environment file found"
        exit 1
    fi
}

run_pre_deployment_checks() {
    print_header "Running Pre-deployment Checks"
    
    # Check if services are healthy
    print_info "Checking service health..."
    
    # Run tests
    if [ "$ENVIRONMENT" = "production" ]; then
        print_warning "Running tests before production deployment..."
        cd "$PROJECT_ROOT"
        
        if python -m pytest tests/unit -v --tb=short > /dev/null 2>&1; then
            print_success "Unit tests passed"
        else
            print_error "Unit tests failed"
            read -p "Continue deployment anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    # Check disk space
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 85 ]; then
        print_warning "Disk usage is high: ${DISK_USAGE}%"
    else
        print_success "Disk usage is acceptable: ${DISK_USAGE}%"
    fi
    
    # Backup current deployment
    if [ "$ENVIRONMENT" = "production" ]; then
        print_warning "Creating backup of current deployment..."
        "$SCRIPT_DIR/backup.sh" --type pre-deployment --environment "$ENVIRONMENT"
        print_success "Backup completed"
    fi
}

build_docker_images() {
    print_header "Building Docker Images"
    
    cd "$PROJECT_ROOT"
    
    if [ -n "$ROLLBACK_VERSION" ]; then
        print_info "Skipping build for rollback deployment"
        return 0
    fi
    
    # Build images
    print_warning "Building ReAgent API image..."
    docker build -t reagent-api:$VERSION -f Dockerfile --target api .
    print_success "Built reagent-api:$VERSION"
    
    print_warning "Building ReAgent Worker image..."
    docker build -t reagent-worker:$VERSION -f Dockerfile --target worker .
    print_success "Built reagent-worker:$VERSION"
    
    # Tag images for registry
    if [ -n "$DOCKER_REGISTRY" ]; then
        docker tag reagent-api:$VERSION "$DOCKER_REGISTRY/reagent-api:$VERSION"
        docker tag reagent-worker:$VERSION "$DOCKER_REGISTRY/reagent-worker:$VERSION"
        
        # Push to registry
        print_warning "Pushing images to registry..."
        docker push "$DOCKER_REGISTRY/reagent-api:$VERSION"
        docker push "$DOCKER_REGISTRY/reagent-worker:$VERSION"
        print_success "Pushed images to registry"
    fi
}

deploy_docker() {
    print_header "Deploying with Docker Compose"
    
    cd "$PROJECT_ROOT"
    
    # Create deployment-specific compose file
    COMPOSE_FILE="docker-compose.$ENVIRONMENT.yml"
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_warning "Creating $COMPOSE_FILE from template..."
        cp docker-compose.yml "$COMPOSE_FILE"
        
        # Update image versions
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|reagent_api:latest|reagent-api:$VERSION|g" "$COMPOSE_FILE"
            sed -i '' "s|reagent_worker:latest|reagent-worker:$VERSION|g" "$COMPOSE_FILE"
        else
            sed -i "s|reagent_api:latest|reagent-api:$VERSION|g" "$COMPOSE_FILE"
            sed -i "s|reagent_worker:latest|reagent-worker:$VERSION|g" "$COMPOSE_FILE"
        fi
    fi
    
    # Stop existing services
    print_warning "Stopping existing services..."
    docker-compose -f "$COMPOSE_FILE" down
    
    # Start new services
    print_warning "Starting services with version $VERSION..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    print_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check service status
    if docker-compose -f "$COMPOSE_FILE" ps | grep -E "reagent.*Up" | grep -q "Up"; then
        print_success "Services are running"
    else
        print_error "Some services failed to start"
        docker-compose -f "$COMPOSE_FILE" ps
        exit 1
    fi
}

deploy_kubernetes() {
    print_header "Deploying to Kubernetes"
    
    cd "$PROJECT_ROOT/k8s"
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    CLUSTER=$(kubectl config current-context)
    print_info "Deploying to cluster: $CLUSTER"
    
    # Create namespace if not exists
    kubectl create namespace $K8S_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    print_success "Namespace $K8S_NAMESPACE ready"
    
    # Update image versions in deployments
    if [ -n "$ROLLBACK_VERSION" ]; then
        VERSION="$ROLLBACK_VERSION"
        print_warning "Rolling back to version $VERSION"
    fi
    
    # Apply configurations
    print_warning "Applying Kubernetes configurations..."
    
    # Update configmap
    kubectl create configmap reagent-config \
        --from-env-file="$PROJECT_ROOT/.env.$ENVIRONMENT" \
        --namespace=$K8S_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply secrets (if exist)
    if [ -f "secrets.$ENVIRONMENT.yaml" ]; then
        kubectl apply -f "secrets.$ENVIRONMENT.yaml" -n $K8S_NAMESPACE
        print_success "Applied secrets"
    fi
    
    # Apply deployments with new version
    for deployment in deployments/*.yaml; do
        # Create temporary file with updated version
        temp_file=$(mktemp)
        sed "s|image: .*reagent-api:.*|image: $DOCKER_REGISTRY/reagent-api:$VERSION|g; \
             s|image: .*reagent-worker:.*|image: $DOCKER_REGISTRY/reagent-worker:$VERSION|g" \
             "$deployment" > "$temp_file"
        
        kubectl apply -f "$temp_file" -n $K8S_NAMESPACE
        rm "$temp_file"
        print_success "Applied $(basename $deployment)"
    done
    
    # Apply services
    kubectl apply -f services/ -n $K8S_NAMESPACE
    print_success "Applied services"
    
    # Wait for rollout
    print_warning "Waiting for deployment rollout..."
    kubectl rollout status deployment/reagent-api -n $K8S_NAMESPACE --timeout=300s
    kubectl rollout status deployment/reagent-worker -n $K8S_NAMESPACE --timeout=300s
    
    # Check pod status
    print_info "Pod status:"
    kubectl get pods -n $K8S_NAMESPACE
}

deploy_docker_swarm() {
    print_header "Deploying to Docker Swarm"
    
    cd "$PROJECT_ROOT"
    
    # Check if swarm is initialized
    if ! docker info | grep -q "Swarm: active"; then
        print_error "Docker Swarm is not initialized"
        exit 1
    fi
    
    # Create stack compose file
    STACK_FILE="docker-stack.$ENVIRONMENT.yml"
    
    if [ ! -f "$STACK_FILE" ]; then
        print_warning "Creating $STACK_FILE..."
        # Convert docker-compose to stack format
        cp docker-compose.yml "$STACK_FILE"
        
        # Add deploy sections
        # This is simplified - in production, you'd have a proper stack file
        print_info "Using docker-compose.yml as stack file"
    fi
    
    # Deploy stack
    print_warning "Deploying stack reagent-$ENVIRONMENT..."
    docker stack deploy -c "$STACK_FILE" "reagent-$ENVIRONMENT"
    
    # Wait for services
    sleep 10
    
    # Check service status
    docker stack services "reagent-$ENVIRONMENT"
    print_success "Stack deployed"
}

run_post_deployment_checks() {
    print_header "Running Post-deployment Checks"
    
    # Wait for services to stabilize
    print_info "Waiting for services to stabilize..."
    sleep 20
    
    # Health checks
    print_warning "Running health checks..."
    
    case $DEPLOYMENT_TYPE in
        docker|docker-swarm)
            # Check API health
            if curl -f http://localhost:8000/health > /dev/null 2>&1; then
                print_success "API health check passed"
            else
                print_error "API health check failed"
            fi
            ;;
        kubernetes)
            # Get service endpoint
            API_ENDPOINT=$(kubectl get service reagent-api -n $K8S_NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
            if [ -z "$API_ENDPOINT" ]; then
                API_ENDPOINT="localhost"
                kubectl port-forward service/reagent-api 8000:8000 -n $K8S_NAMESPACE &
                PF_PID=$!
                sleep 5
            fi
            
            if curl -f http://$API_ENDPOINT:8000/health > /dev/null 2>&1; then
                print_success "API health check passed"
            else
                print_error "API health check failed"
            fi
            
            [ -n "$PF_PID" ] && kill $PF_PID 2>/dev/null
            ;;
    esac
    
    # Run smoke tests
    if [ "$ENVIRONMENT" = "production" ]; then
        print_warning "Running smoke tests..."
        
        # Simple question test
        RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/questions \
            -H "Content-Type: application/json" \
            -d '{"question": "What is 2+2?"}' || echo '{"error": "Failed"}')
        
        if echo "$RESPONSE" | grep -q "task_id"; then
            print_success "Smoke test passed"
        else
            print_error "Smoke test failed: $RESPONSE"
        fi
    fi
    
    # Check metrics
    if curl -f http://localhost:9090/api/v1/query?query=up > /dev/null 2>&1; then
        print_success "Prometheus is accessible"
    else
        print_warning "Prometheus is not accessible"
    fi
}

update_load_balancer() {
    print_header "Updating Load Balancer"
    
    if [ "$ENVIRONMENT" != "production" ]; then
        print_info "Skipping load balancer update for $ENVIRONMENT"
        return 0
    fi
    
    # This is environment-specific
    # Example for AWS ALB, Nginx, HAProxy, etc.
    print_warning "Load balancer update not implemented"
    print_info "Please update your load balancer configuration manually"
}

cleanup_old_deployments() {
    print_header "Cleaning Up Old Deployments"
    
    case $DEPLOYMENT_TYPE in
        docker)
            # Remove old containers
            print_warning "Removing stopped containers..."
            docker container prune -f
            
            # Remove old images (keep last 3 versions)
            print_warning "Cleaning old images..."
            docker images | grep reagent | tail -n +4 | awk '{print $3}' | xargs -r docker rmi -f || true
            ;;
        kubernetes)
            # Clean up old replica sets
            print_warning "Cleaning old replica sets..."
            kubectl delete replicaset -n $K8S_NAMESPACE \
                $(kubectl get replicaset -n $K8S_NAMESPACE -o jsonpath='{.items[?(@.spec.replicas==0)].metadata.name}') \
                2>/dev/null || true
            ;;
    esac
    
    print_success "Cleanup completed"
}

send_deployment_notification() {
    print_header "Sending Deployment Notification"
    
    DEPLOY_INFO="Environment: $ENVIRONMENT
Type: $DEPLOYMENT_TYPE
Version: ${VERSION:-$ROLLBACK_VERSION}
Timestamp: $(date)
Status: Success"
    
    # Slack notification (if configured)
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-type: application/json' \
            -d "{\"text\": \"ReAgent Deployment Completed\n\`\`\`$DEPLOY_INFO\`\`\`\"}" \
            > /dev/null 2>&1 || true
    fi
    
    # Email notification (if configured)
    if [ -n "${DEPLOYMENT_EMAIL:-}" ]; then
        echo "$DEPLOY_INFO" | mail -s "ReAgent Deployment - $ENVIRONMENT" "$DEPLOYMENT_EMAIL" || true
    fi
    
    print_success "Notifications sent"
}

rollback_deployment() {
    print_header "Rolling Back Deployment"
    
    print_warning "Rolling back to version: $ROLLBACK_VERSION"
    
    case $DEPLOYMENT_TYPE in
        docker)
            # Simply redeploy with old version
            VERSION="$ROLLBACK_VERSION"
            deploy_docker
            ;;
        kubernetes)
            # Use kubectl rollout undo or redeploy with old version
            print_warning "Rolling back Kubernetes deployment..."
            kubectl rollout undo deployment/reagent-api -n $K8S_NAMESPACE
            kubectl rollout undo deployment/reagent-worker -n $K8S_NAMESPACE
            
            # Wait for rollout
            kubectl rollout status deployment/reagent-api -n $K8S_NAMESPACE
            kubectl rollout status deployment/reagent-worker -n $K8S_NAMESPACE
            ;;
        docker-swarm)
            # Update service with previous version
            docker service update --image "reagent-api:$ROLLBACK_VERSION" "reagent-${ENVIRONMENT}_reagent_api"
            docker service update --image "reagent-worker:$ROLLBACK_VERSION" "reagent-${ENVIRONMENT}_reagent_worker"
            ;;
    esac
    
    print_success "Rollback completed"
}

# Main execution
main() {
    # Create log directory
    mkdir -p "$(dirname "$DEPLOYMENT_LOG")"
    
    print_header "ReAgent Deployment Script"
    print_info "Deployment started at $(date)"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Validate environment
    validate_environment
    
    # Load environment configuration
    load_environment_config
    
    # Check if rollback
    if [ -n "$ROLLBACK_VERSION" ]; then
        rollback_deployment
        run_post_deployment_checks
        send_deployment_notification
        print_success "Rollback completed successfully!"
        exit 0
    fi
    
    # Normal deployment flow
    run_pre_deployment_checks
    build_docker_images
    
    # Deploy based on type
    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        docker-swarm)
            deploy_docker_swarm
            ;;
    esac
    
    # Post-deployment tasks
    run_post_deployment_checks
    update_load_balancer
    cleanup_old_deployments
    send_deployment_notification
    
    print_success "Deployment completed successfully!"
    print_info "Deployment log saved to: $DEPLOYMENT_LOG"
}

# Error handler
trap 'print_error "Deployment failed!"; exit 1' ERR

# Run main function
main "$@"
