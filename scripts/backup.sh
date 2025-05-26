#!/bin/bash

# ReAgent System Backup Script
# This script handles backups of the ReAgent system including database, configurations, and data

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
BACKUP_ROOT="${BACKUP_ROOT:-$PROJECT_ROOT/backups}"
BACKUP_DIR="$BACKUP_ROOT/backup_$TIMESTAMP"
BACKUP_LOG="$BACKUP_DIR/backup.log"

# Default values
BACKUP_TYPE="full"
ENVIRONMENT="production"
COMPONENTS=()
RETENTION_DAYS=30
COMPRESS=true
ENCRYPT=false
UPLOAD_S3=false
S3_BUCKET=""
RESTORE_FROM=""

# Functions
print_header() {
    echo -e "${BLUE}===============================================${NC}" | tee -a "$BACKUP_LOG"
    echo -e "${BLUE}$1${NC}" | tee -a "$BACKUP_LOG"
    echo -e "${BLUE}===============================================${NC}" | tee -a "$BACKUP_LOG"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$BACKUP_LOG"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "$BACKUP_LOG"
}

print_error() {
    echo -e "${RED}✗ $1${NC}" | tee -a "$BACKUP_LOG"
}

print_info() {
    echo -e "${PURPLE}ℹ $1${NC}" | tee -a "$BACKUP_LOG"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Backup Options:"
    echo "  -t, --type TYPE          Backup type (full/incremental/database/config)"
    echo "  -e, --environment ENV    Environment (development/staging/production)"
    echo "  -c, --components COMP    Specific components to backup (comma-separated)"
    echo "                          Options: postgres,redis,elasticsearch,config,logs,metrics"
    echo "  --no-compress           Don't compress backup"
    echo "  --encrypt               Encrypt backup (requires GPG key)"
    echo "  --s3 BUCKET            Upload to S3 bucket"
    echo "  --retention DAYS        Keep backups for N days (default: 30)"
    echo
    echo "Restore Options:"
    echo "  --restore PATH          Restore from backup path"
    echo "  --restore-component COMP Restore specific component"
    echo
    echo "Examples:"
    echo "  $0 -t full -e production"
    echo "  $0 -t database -c postgres,redis"
    echo "  $0 -t full --encrypt --s3 my-backup-bucket"
    echo "  $0 --restore /path/to/backup_20240115_120000"
    exit 1
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                BACKUP_TYPE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -c|--components)
                IFS=',' read -ra COMPONENTS <<< "$2"
                shift 2
                ;;
            --no-compress)
                COMPRESS=false
                shift
                ;;
            --encrypt)
                ENCRYPT=true
                shift
                ;;
            --s3)
                UPLOAD_S3=true
                S3_BUCKET="$2"
                shift 2
                ;;
            --retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            --restore)
                RESTORE_FROM="$2"
                shift 2
                ;;
            --restore-component)
                RESTORE_COMPONENT="$2"
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
}

validate_environment() {
    print_header "Validating Environment"
    
    # Check if restoring
    if [ -n "$RESTORE_FROM" ]; then
        if [ ! -d "$RESTORE_FROM" ] && [ ! -f "$RESTORE_FROM" ]; then
            print_error "Restore path does not exist: $RESTORE_FROM"
            exit 1
        fi
        print_success "Restore from: $RESTORE_FROM"
        return 0
    fi
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Initialize log file
    touch "$BACKUP_LOG"
    
    print_success "Backup directory: $BACKUP_DIR"
    print_success "Environment: $ENVIRONMENT"
    print_success "Backup type: $BACKUP_TYPE"
    
    # Set default components based on backup type
    if [ ${#COMPONENTS[@]} -eq 0 ]; then
        case $BACKUP_TYPE in
            full)
                COMPONENTS=("postgres" "redis" "elasticsearch" "config" "logs" "metrics")
                ;;
            database)
                COMPONENTS=("postgres" "redis" "elasticsearch")
                ;;
            config)
                COMPONENTS=("config")
                ;;
            incremental)
                COMPONENTS=("postgres" "logs")
                ;;
        esac
    fi
    
    print_info "Components to backup: ${COMPONENTS[*]}"
}

backup_postgres() {
    print_header "Backing up PostgreSQL"
    
    DB_BACKUP_DIR="$BACKUP_DIR/postgres"
    mkdir -p "$DB_BACKUP_DIR"
    
    # Get database credentials from environment
    DB_HOST="${POSTGRES_HOST:-localhost}"
    DB_PORT="${POSTGRES_PORT:-5432}"
    DB_NAME="${POSTGRES_DB:-reagent_db}"
    DB_USER="${POSTGRES_USER:-reagent_user}"
    DB_PASS="${POSTGRES_PASSWORD:-}"
    
    # Check if running in Docker
    if docker ps | grep -q postgres; then
        print_info "Backing up PostgreSQL from Docker container"
        
        # Dump database
        docker exec postgres pg_dump \
            -U "$DB_USER" \
            -d "$DB_NAME" \
            --verbose \
            --no-owner \
            --no-acl \
            --format=custom \
            --file="/tmp/reagent_backup.dump"
        
        # Copy dump from container
        docker cp postgres:/tmp/reagent_backup.dump "$DB_BACKUP_DIR/reagent_db.dump"
        
        # Clean up
        docker exec postgres rm /tmp/reagent_backup.dump
        
        # Also backup globals
        docker exec postgres pg_dumpall \
            -U "$DB_USER" \
            --globals-only \
            --file="/tmp/globals.sql"
        
        docker cp postgres:/tmp/globals.sql "$DB_BACKUP_DIR/globals.sql"
        docker exec postgres rm /tmp/globals.sql
        
    else
        print_info "Backing up PostgreSQL from host"
        
        # Set password in environment
        export PGPASSWORD="$DB_PASS"
        
        # Dump database
        pg_dump \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            -d "$DB_NAME" \
            --verbose \
            --no-owner \
            --no-acl \
            --format=custom \
            --file="$DB_BACKUP_DIR/reagent_db.dump"
        
        # Backup globals
        pg_dumpall \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            --globals-only \
            --file="$DB_BACKUP_DIR/globals.sql"
        
        unset PGPASSWORD
    fi
    
    # Get database size
    DB_SIZE=$(du -sh "$DB_BACKUP_DIR" | cut -f1)
    print_success "PostgreSQL backup completed (Size: $DB_SIZE)"
    
    # Save metadata
    cat > "$DB_BACKUP_DIR/metadata.json" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "database": "$DB_NAME",
    "host": "$DB_HOST",
    "size": "$DB_SIZE",
    "format": "custom",
    "version": "$(docker exec postgres psql --version 2>/dev/null | head -n1 || psql --version | head -n1)"
}
EOF
}

backup_redis() {
    print_header "Backing up Redis"
    
    REDIS_BACKUP_DIR="$BACKUP_DIR/redis"
    mkdir -p "$REDIS_BACKUP_DIR"
    
    REDIS_HOST="${REDIS_HOST:-localhost}"
    REDIS_PORT="${REDIS_PORT:-6379}"
    
    # Check if running in Docker
    if docker ps | grep -q redis; then
        print_info "Backing up Redis from Docker container"
        
        # Trigger BGSAVE
        docker exec redis redis-cli BGSAVE
        
        # Wait for background save to complete
        print_info "Waiting for Redis background save..."
        while [ "$(docker exec redis redis-cli LASTSAVE)" = "$(docker exec redis redis-cli LASTSAVE)" ]; do
            sleep 1
        done
        
        # Copy dump file
        docker cp redis:/data/dump.rdb "$REDIS_BACKUP_DIR/dump.rdb"
        
        # Also backup AOF if enabled
        if docker exec redis test -f /data/appendonly.aof; then
            docker cp redis:/data/appendonly.aof "$REDIS_BACKUP_DIR/appendonly.aof"
            print_success "Backed up Redis AOF file"
        fi
        
    else
        print_info "Backing up Redis from host"
        
        # Trigger BGSAVE
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE
        
        # Wait for background save
        print_info "Waiting for Redis background save..."
        LAST_SAVE=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE)
        while [ "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE)" = "$LAST_SAVE" ]; do
            sleep 1
        done
        
        # Find and copy dump file
        REDIS_DIR=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG GET dir | tail -1)
        cp "$REDIS_DIR/dump.rdb" "$REDIS_BACKUP_DIR/dump.rdb"
    fi
    
    # Get backup size
    REDIS_SIZE=$(du -sh "$REDIS_BACKUP_DIR" | cut -f1)
    print_success "Redis backup completed (Size: $REDIS_SIZE)"
}

backup_elasticsearch() {
    print_header "Backing up Elasticsearch"
    
    ES_BACKUP_DIR="$BACKUP_DIR/elasticsearch"
    mkdir -p "$ES_BACKUP_DIR"
    
    ES_HOST="${ELASTICSEARCH_HOST:-localhost}"
    ES_PORT="${ELASTICSEARCH_PORT:-9200}"
    
    # Create snapshot repository
    print_info "Creating snapshot repository..."
    curl -X PUT "http://$ES_HOST:$ES_PORT/_snapshot/backup_repo" \
        -H 'Content-Type: application/json' \
        -d "{
            \"type\": \"fs\",
            \"settings\": {
                \"location\": \"/tmp/es_backup\",
                \"compress\": true
            }
        }" 2>/dev/null || true
    
    # Create snapshot
    SNAPSHOT_NAME="snapshot_$TIMESTAMP"
    print_info "Creating Elasticsearch snapshot: $SNAPSHOT_NAME"
    
    curl -X PUT "http://$ES_HOST:$ES_PORT/_snapshot/backup_repo/$SNAPSHOT_NAME?wait_for_completion=true" \
        -H 'Content-Type: application/json' \
        -d '{
            "indices": "reagent-*",
            "include_global_state": false
        }' > "$ES_BACKUP_DIR/snapshot_response.json" 2>/dev/null
    
    # Export snapshot files
    if docker ps | grep -q elasticsearch; then
        docker exec elasticsearch tar -czf /tmp/es_backup.tar.gz /tmp/es_backup
        docker cp elasticsearch:/tmp/es_backup.tar.gz "$ES_BACKUP_DIR/snapshot.tar.gz"
        docker exec elasticsearch rm -rf /tmp/es_backup /tmp/es_backup.tar.gz
    fi
    
    # Also export index mappings
    print_info "Exporting index mappings..."
    curl -X GET "http://$ES_HOST:$ES_PORT/reagent-*/_mapping" \
        > "$ES_BACKUP_DIR/mappings.json" 2>/dev/null
    
    print_success "Elasticsearch backup completed"
}

backup_config() {
    print_header "Backing up Configuration Files"
    
    CONFIG_BACKUP_DIR="$BACKUP_DIR/config"
    mkdir -p "$CONFIG_BACKUP_DIR"
    
    # List of configuration files to backup
    CONFIG_FILES=(
        ".env"
        ".env.$ENVIRONMENT"
        "docker-compose.yml"
        "docker-compose.$ENVIRONMENT.yml"
        "configs/prompts.yaml"
        "configs/redis.conf"
        "configs/nginx.conf"
        "monitoring/prometheus.yml"
        "monitoring/alerting/rules.yml"
        "monitoring/grafana/dashboards/*.json"
        "k8s/*.yaml"
    )
    
    # Copy configuration files
    for config in "${CONFIG_FILES[@]}"; do
        if [ -e "$PROJECT_ROOT/$config" ]; then
            # Create directory structure
            dir=$(dirname "$config")
            mkdir -p "$CONFIG_BACKUP_DIR/$dir"
            
            # Copy file(s)
            cp -r "$PROJECT_ROOT/$config" "$CONFIG_BACKUP_DIR/$config" 2>/dev/null || true
            print_success "Backed up $config"
        fi
    done
    
    # Remove sensitive data from .env files
    if [ "$ENCRYPT" = false ]; then
        print_warning "Sanitizing sensitive data in .env files"
        find "$CONFIG_BACKUP_DIR" -name ".env*" -type f -exec sed -i.bak \
            -e 's/\(PASSWORD=\).*/\1[REDACTED]/' \
            -e 's/\(API_KEY=\).*/\1[REDACTED]/' \
            -e 's/\(SECRET=\).*/\1[REDACTED]/' \
            -e 's/\(TOKEN=\).*/\1[REDACTED]/' {} \;
        find "$CONFIG_BACKUP_DIR" -name "*.bak" -delete
    fi
    
    print_success "Configuration backup completed"
}

backup_logs() {
    print_header "Backing up Logs"
    
    LOGS_BACKUP_DIR="$BACKUP_DIR/logs"
    mkdir -p "$LOGS_BACKUP_DIR"
    
    # Backup application logs
    if [ -d "$PROJECT_ROOT/logs" ]; then
        print_info "Backing up application logs..."
        
        # For incremental backup, only copy recent logs
        if [ "$BACKUP_TYPE" = "incremental" ]; then
            find "$PROJECT_ROOT/logs" -type f -mtime -1 -exec cp {} "$LOGS_BACKUP_DIR/" \;
        else
            cp -r "$PROJECT_ROOT/logs/"* "$LOGS_BACKUP_DIR/" 2>/dev/null || true
        fi
    fi
    
    # Backup Docker logs
    print_info "Backing up Docker container logs..."
    for container in $(docker ps --format "{{.Names}}" | grep reagent); do
        docker logs "$container" > "$LOGS_BACKUP_DIR/${container}.log" 2>&1 || true
    done
    
    # Compress logs
    if [ "$COMPRESS" = true ]; then
        print_info "Compressing log files..."
        find "$LOGS_BACKUP_DIR" -name "*.log" -exec gzip {} \;
    fi
    
    LOGS_SIZE=$(du -sh "$LOGS_BACKUP_DIR" | cut -f1)
    print_success "Logs backup completed (Size: $LOGS_SIZE)"
}

backup_metrics() {
    print_header "Backing up Metrics Data"
    
    METRICS_BACKUP_DIR="$BACKUP_DIR/metrics"
    mkdir -p "$METRICS_BACKUP_DIR"
    
    # Backup Prometheus data
    if docker ps | grep -q prometheus; then
        print_info "Creating Prometheus snapshot..."
        
        # Trigger snapshot via API
        SNAPSHOT_RESPONSE=$(curl -X POST http://localhost:9090/api/v1/admin/tsdb/snapshot 2>/dev/null)
        SNAPSHOT_NAME=$(echo "$SNAPSHOT_RESPONSE" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
        
        if [ -n "$SNAPSHOT_NAME" ]; then
            # Copy snapshot
            docker cp "prometheus:/prometheus/snapshots/$SNAPSHOT_NAME" "$METRICS_BACKUP_DIR/prometheus_snapshot"
            print_success "Prometheus snapshot created: $SNAPSHOT_NAME"
        else
            print_warning "Failed to create Prometheus snapshot"
        fi
    fi
    
    # Export Grafana dashboards
    if docker ps | grep -q grafana; then
        print_info "Exporting Grafana dashboards..."
        mkdir -p "$METRICS_BACKUP_DIR/grafana"
        
        # This requires Grafana API access
        # Simplified version - in production, use Grafana API
        if [ -d "$PROJECT_ROOT/monitoring/grafana/dashboards" ]; then
            cp -r "$PROJECT_ROOT/monitoring/grafana/dashboards" "$METRICS_BACKUP_DIR/grafana/"
        fi
    fi
    
    print_success "Metrics backup completed"
}

compress_backup() {
    if [ "$COMPRESS" = false ]; then
        return 0
    fi
    
    print_header "Compressing Backup"
    
    cd "$BACKUP_ROOT"
    ARCHIVE_NAME="backup_${TIMESTAMP}.tar.gz"
    
    print_info "Creating archive: $ARCHIVE_NAME"
    tar -czf "$ARCHIVE_NAME" "backup_$TIMESTAMP"
    
    # Get compressed size
    COMPRESSED_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)
    print_success "Backup compressed (Size: $COMPRESSED_SIZE)"
    
    # Remove uncompressed backup
    rm -rf "backup_$TIMESTAMP"
    
    # Update backup directory for encryption/upload
    BACKUP_DIR="$BACKUP_ROOT/$ARCHIVE_NAME"
}

encrypt_backup() {
    if [ "$ENCRYPT" = false ]; then
        return 0
    fi
    
    print_header "Encrypting Backup"
    
    # Check for GPG
    if ! command -v gpg &> /dev/null; then
        print_error "GPG is not installed"
        return 1
    fi
    
    # Get GPG recipient
    read -p "Enter GPG key ID or email for encryption: " GPG_RECIPIENT
    
    if [ -z "$GPG_RECIPIENT" ]; then
        print_error "GPG recipient required for encryption"
        return 1
    fi
    
    # Encrypt the backup
    print_info "Encrypting backup for recipient: $GPG_RECIPIENT"
    
    if [ -f "$BACKUP_DIR" ]; then
        # Single file (compressed archive)
        gpg --encrypt --recipient "$GPG_RECIPIENT" "$BACKUP_DIR"
        rm "$BACKUP_DIR"
        BACKUP_DIR="${BACKUP_DIR}.gpg"
    else
        # Directory - create encrypted archive
        cd "$BACKUP_ROOT"
        tar -czf - "backup_$TIMESTAMP" | gpg --encrypt --recipient "$GPG_RECIPIENT" > "backup_${TIMESTAMP}.tar.gz.gpg"
        rm -rf "backup_$TIMESTAMP"
        BACKUP_DIR="$BACKUP_ROOT/backup_${TIMESTAMP}.tar.gz.gpg"
    fi
    
    print_success "Backup encrypted"
}

upload_to_s3() {
    if [ "$UPLOAD_S3" = false ]; then
        return 0
    fi
    
    print_header "Uploading to S3"
    
    # Check for AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed"
        return 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured"
        return 1
    fi
    
    # Upload to S3
    S3_PATH="s3://$S3_BUCKET/reagent-backups/$ENVIRONMENT/$(basename "$BACKUP_DIR")"
    
    print_info "Uploading to: $S3_PATH"
    aws s3 cp "$BACKUP_DIR" "$S3_PATH" --storage-class STANDARD_IA
    
    print_success "Backup uploaded to S3"
    
    # Create lifecycle policy for automatic deletion
    cat > /tmp/lifecycle.json <<EOF
{
    "Rules": [{
        "ID": "DeleteOldBackups",
        "Status": "Enabled",
        "Prefix": "reagent-backups/",
        "Expiration": {
            "Days": $RETENTION_DAYS
        }
    }]
}
EOF
    
    aws s3api put-bucket-lifecycle-configuration \
        --bucket "$S3_BUCKET" \
        --lifecycle-configuration file:///tmp/lifecycle.json 2>/dev/null || true
    
    rm /tmp/lifecycle.json
}

cleanup_old_backups() {
    print_header "Cleaning Up Old Backups"
    
    # Clean local backups
    print_info "Removing backups older than $RETENTION_DAYS days"
    
    find "$BACKUP_ROOT" -name "backup_*" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    find "$BACKUP_ROOT" -name "backup_*" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
    
    REMAINING=$(ls -1 "$BACKUP_ROOT" | grep -c "backup_" || echo "0")
    print_success "Cleanup completed. Remaining backups: $REMAINING"
}

create_backup_report() {
    print_header "Creating Backup Report"
    
    REPORT_FILE="$BACKUP_ROOT/backup_report_$TIMESTAMP.txt"
    
    cat > "$REPORT_FILE" <<EOF
ReAgent Backup Report
====================
Timestamp: $(date)
Environment: $ENVIRONMENT
Backup Type: $BACKUP_TYPE
Components: ${COMPONENTS[*]}

Backup Location: $BACKUP_DIR
Compressed: $COMPRESS
Encrypted: $ENCRYPT
Uploaded to S3: $UPLOAD_S3

Backup Contents:
EOF
    
    # Add directory listing
    if [ -d "$BACKUP_DIR" ]; then
        echo -e "\nDirectory Structure:" >> "$REPORT_FILE"
        tree "$BACKUP_DIR" >> "$REPORT_FILE" 2>/dev/null || ls -la "$BACKUP_DIR" >> "$REPORT_FILE"
    elif [ -f "$BACKUP_DIR" ]; then
        echo -e "\nBackup File:" >> "$REPORT_FILE"
        ls -lh "$BACKUP_DIR" >> "$REPORT_FILE"
    fi
    
    # Add sizes
    echo -e "\nBackup Sizes:" >> "$REPORT_FILE"
    du -sh "$BACKUP_ROOT/backup_$TIMESTAMP"* >> "$REPORT_FILE" 2>/dev/null || true
    
    print_success "Backup report created: $REPORT_FILE"
}

restore_backup() {
    print_header "Restoring from Backup"
    
    if [ ! -d "$RESTORE_FROM" ] && [ ! -f "$RESTORE_FROM" ]; then
        print_error "Restore path does not exist: $RESTORE_FROM"
        exit 1
    fi
    
    # Create temporary directory for extraction
    TEMP_RESTORE="/tmp/reagent_restore_$$"
    mkdir -p "$TEMP_RESTORE"
    
    # Extract backup if compressed/encrypted
    if [[ "$RESTORE_FROM" == *.gpg ]]; then
        print_info "Decrypting backup..."
        gpg --decrypt "$RESTORE_FROM" | tar -xzf - -C "$TEMP_RESTORE"
        RESTORE_FROM="$TEMP_RESTORE/$(ls "$TEMP_RESTORE" | head -1)"
    elif [[ "$RESTORE_FROM" == *.tar.gz ]]; then
        print_info "Extracting backup..."
        tar -xzf "$RESTORE_FROM" -C "$TEMP_RESTORE"
        RESTORE_FROM="$TEMP_RESTORE/$(ls "$TEMP_RESTORE" | head -1)"
    fi
    
    # Confirm restoration
    echo -e "${YELLOW}WARNING: This will restore data from:${NC}"
    echo -e "${YELLOW}$RESTORE_FROM${NC}"
    echo -e "${YELLOW}This may overwrite existing data!${NC}"
    read -p "Continue with restoration? (yes/NO): " CONFIRM
    
    if [ "$CONFIRM" != "yes" ]; then
        print_warning "Restoration cancelled"
        rm -rf "$TEMP_RESTORE"
        exit 0
    fi
    
    # Stop services before restoration
    print_warning "Stopping services..."
    cd "$PROJECT_ROOT"
    docker-compose down
    
    # Restore each component
    if [ -d "$RESTORE_FROM/postgres" ]; then
        restore_postgres "$RESTORE_FROM/postgres"
    fi
    
    if [ -d "$RESTORE_FROM/redis" ]; then
        restore_redis "$RESTORE_FROM/redis"
    fi
    
    if [ -d "$RESTORE_FROM/elasticsearch" ]; then
        restore_elasticsearch "$RESTORE_FROM/elasticsearch"
    fi
    
    if [ -d "$RESTORE_FROM/config" ]; then
        restore_config "$RESTORE_FROM/config"
    fi
    
    # Start services
    print_warning "Starting services..."
    docker-compose up -d
    
    # Wait for services
    print_info "Waiting for services to be ready..."
    sleep 30
    
    # Verify restoration
    verify_restoration
    
    # Cleanup
    rm -rf "$TEMP_RESTORE"
    
    print_success "Restoration completed!"
}

restore_postgres() {
    local POSTGRES_RESTORE_DIR="$1"
    print_header "Restoring PostgreSQL"
    
    # Start only PostgreSQL
    docker-compose up -d postgres
    sleep 10
    
    # Wait for PostgreSQL to be ready
    while ! docker exec postgres pg_isready -U reagent_user > /dev/null 2>&1; do
        print_info "Waiting for PostgreSQL..."
        sleep 2
    done
    
    # Drop existing database and recreate
    print_warning "Dropping existing database..."
    docker exec postgres psql -U reagent_user -d postgres -c "DROP DATABASE IF EXISTS reagent_db;"
    docker exec postgres psql -U reagent_user -d postgres -c "CREATE DATABASE reagent_db;"
    
    # Copy backup file to container
    docker cp "$POSTGRES_RESTORE_DIR/reagent_db.dump" postgres:/tmp/restore.dump
    
    # Restore database
    print_info "Restoring database..."
    docker exec postgres pg_restore \
        -U reagent_user \
        -d reagent_db \
        --verbose \
        --no-owner \
        --no-acl \
        /tmp/restore.dump
    
    # Clean up
    docker exec postgres rm /tmp/restore.dump
    
    print_success "PostgreSQL restored"
}

restore_redis() {
    local REDIS_RESTORE_DIR="$1"
    print_header "Restoring Redis"
    
    # Stop Redis first
    docker-compose stop redis
    
    # Copy dump file
    docker cp "$REDIS_RESTORE_DIR/dump.rdb" redis:/data/dump.rdb
    
    # Start Redis
    docker-compose up -d redis
    
    print_success "Redis restored"
}

restore_elasticsearch() {
    local ES_RESTORE_DIR="$1"
    print_header "Restoring Elasticsearch"
    
    # Start Elasticsearch
    docker-compose up -d elasticsearch
    sleep 20
    
    # Wait for Elasticsearch
    while ! curl -s http://localhost:9200/_cluster/health > /dev/null; do
        print_info "Waiting for Elasticsearch..."
        sleep 5
    done
    
    # Extract snapshot if exists
    if [ -f "$ES_RESTORE_DIR/snapshot.tar.gz" ]; then
        print_info "Extracting snapshot..."
        docker exec elasticsearch mkdir -p /tmp/es_restore
        docker cp "$ES_RESTORE_DIR/snapshot.tar.gz" elasticsearch:/tmp/
        docker exec elasticsearch tar -xzf /tmp/snapshot.tar.gz -C /tmp/es_restore
        
        # Register repository
        curl -X PUT "http://localhost:9200/_snapshot/restore_repo" \
            -H 'Content-Type: application/json' \
            -d '{
                "type": "fs",
                "settings": {
                    "location": "/tmp/es_restore/tmp/es_backup"
                }
            }'
        
        # List snapshots and restore
        SNAPSHOTS=$(curl -s "http://localhost:9200/_snapshot/restore_repo/_all" | grep -o '"snapshot":"[^"]*"' | cut -d'"' -f4)
        
        for snapshot in $SNAPSHOTS; do
            print_info "Restoring snapshot: $snapshot"
            curl -X POST "http://localhost:9200/_snapshot/restore_repo/$snapshot/_restore?wait_for_completion=true"
        done
    fi
    
    print_success "Elasticsearch restored"
}

restore_config() {
    local CONFIG_RESTORE_DIR="$1"
    print_header "Restoring Configuration"
    
    # Backup current configs
    print_info "Backing up current configuration..."
    for config in .env configs monitoring/prometheus.yml monitoring/alerting k8s; do
        if [ -e "$PROJECT_ROOT/$config" ]; then
            mv "$PROJECT_ROOT/$config" "$PROJECT_ROOT/${config}.backup.$TIMESTAMP"
        fi
    done
    
    # Restore configs
    print_info "Restoring configuration files..."
    cp -r "$CONFIG_RESTORE_DIR/"* "$PROJECT_ROOT/"
    
    print_success "Configuration restored"
}

verify_restoration() {
    print_header "Verifying Restoration"
    
    # Check PostgreSQL
    if docker exec postgres psql -U reagent_user -d reagent_db -c "SELECT 1;" > /dev/null 2>&1; then
        print_success "PostgreSQL is accessible"
    else
        print_error "PostgreSQL verification failed"
    fi
    
    # Check Redis
    if docker exec redis redis-cli ping | grep -q PONG; then
        print_success "Redis is accessible"
    else
        print_error "Redis verification failed"
    fi
    
    # Check Elasticsearch
    if curl -s http://localhost:9200/_cluster/health | grep -q '"status":"green"'; then
        print_success "Elasticsearch is healthy"
    elif curl -s http://localhost:9200/_cluster/health | grep -q '"status":"yellow"'; then
        print_warning "Elasticsearch is accessible but status is yellow"
    else
        print_error "Elasticsearch verification failed"
    fi
}

# Main execution
main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Validate environment
    validate_environment
    
    # Check if restoring
    if [ -n "$RESTORE_FROM" ]; then
        restore_backup
        exit 0
    fi
    
    # Create backup directory
    mkdir -p "$BACKUP_ROOT"
    
    print_header "Starting ReAgent Backup"
    print_info "Backup started at $(date)"
    
    # Backup each component
    for component in "${COMPONENTS[@]}"; do
        case $component in
            postgres)
                backup_postgres
                ;;
            redis)
                backup_redis
                ;;
            elasticsearch)
                backup_elasticsearch
                ;;
            config)
                backup_config
                ;;
            logs)
                backup_logs
                ;;
            metrics)
                backup_metrics
                ;;
            *)
                print_warning "Unknown component: $component"
                ;;
        esac
    done
    
    # Post-backup operations
    compress_backup
    encrypt_backup
    upload_to_s3
    create_backup_report
    cleanup_old_backups
    
    print_success "Backup completed successfully!"
    print_info "Backup location: $BACKUP_DIR"
    
    # Show summary
    if [ -f "$BACKUP_DIR" ]; then
        print_info "Backup size: $(ls -lh "$BACKUP_DIR" | awk '{print $5}')"
    else
        print_info "Total backup size: $(du -sh "$BACKUP_DIR" | cut -f1)"
    fi
}

# Error handler
trap 'print_error "Backup failed!"; exit 1' ERR

# Run main function
main "$@"