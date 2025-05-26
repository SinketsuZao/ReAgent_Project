-- ReAgent Database Initialization Script
-- This script sets up the PostgreSQL database schema for the ReAgent system

-- Create database if not exists (run as superuser)
-- CREATE DATABASE reagent_db;

-- Connect to the database
\c reagent_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create custom types
CREATE TYPE task_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');
CREATE TYPE question_status AS ENUM ('received', 'decomposed', 'retrieving', 'verifying', 'assembling', 'backtracking', 'completed', 'failed', 'timeout');
CREATE TYPE agent_type AS ENUM ('decomposer', 'retriever', 'verifier', 'assembler', 'supervisor', 'controller');
CREATE TYPE backtracking_scope AS ENUM ('local', 'global');
CREATE TYPE message_type AS ENUM ('assert', 'inform', 'challenge', 'reject', 'accept');

-- Create schema for better organization
CREATE SCHEMA IF NOT EXISTS reagent;
SET search_path TO reagent, public;

-- Questions table
CREATE TABLE IF NOT EXISTS questions (
    id VARCHAR(255) PRIMARY KEY,
    question_text TEXT NOT NULL,
    task_id VARCHAR(255),
    status question_status NOT NULL DEFAULT 'received',
    answer TEXT,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    processing_time FLOAT,
    token_usage INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    error TEXT,
    parent_question_id VARCHAR(255) REFERENCES questions(id) ON DELETE CASCADE,
    INDEX idx_questions_status (status),
    INDEX idx_questions_created (created_at),
    INDEX idx_questions_task (task_id)
);

-- Create full-text search index on question text
CREATE INDEX idx_questions_text_search ON questions USING gin(to_tsvector('english', question_text));

-- Sub-questions table
CREATE TABLE IF NOT EXISTS sub_questions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_question_id VARCHAR(255) NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    sub_question_text TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    answer TEXT,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    dependencies JSONB DEFAULT '[]',
    UNIQUE(parent_question_id, order_index)
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id VARCHAR(255) PRIMARY KEY,
    task_type VARCHAR(50) NOT NULL,
    status task_status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    result JSONB,
    error TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    metadata JSONB DEFAULT '{}',
    INDEX idx_tasks_status (status),
    INDEX idx_tasks_type (task_type),
    INDEX idx_tasks_created (created_at)
);

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    id VARCHAR(50) PRIMARY KEY,
    agent_type agent_type NOT NULL,
    status VARCHAR(20) DEFAULT 'idle',
    total_tasks_processed INTEGER DEFAULT 0,
    total_tasks_failed INTEGER DEFAULT 0,
    total_tokens_used BIGINT DEFAULT 0,
    avg_processing_time FLOAT DEFAULT 0,
    reliability_score FLOAT DEFAULT 1.0 CHECK (reliability_score >= 0 AND reliability_score <= 1),
    last_active TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    configuration JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}'
);

-- Insert default agents
INSERT INTO agents (id, agent_type) VALUES
    ('A_Q', 'decomposer'),
    ('A_R', 'retriever'),
    ('A_V', 'verifier'),
    ('A_A', 'assembler'),
    ('A_S', 'supervisor'),
    ('A_C', 'controller')
ON CONFLICT (id) DO NOTHING;

-- Agent messages table
CREATE TABLE IF NOT EXISTS agent_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id VARCHAR(255) UNIQUE NOT NULL,
    sender_agent_id VARCHAR(50) NOT NULL REFERENCES agents(id),
    message_type message_type NOT NULL,
    content JSONB NOT NULL,
    question_id VARCHAR(255) REFERENCES questions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_messages_sender (sender_agent_id),
    INDEX idx_messages_question (question_id),
    INDEX idx_messages_created (created_at),
    INDEX idx_messages_type (message_type)
);

-- Knowledge assertions table
CREATE TABLE IF NOT EXISTS knowledge_assertions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(50) NOT NULL REFERENCES agents(id),
    question_id VARCHAR(255) REFERENCES questions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    source VARCHAR(255),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    invalidated_at TIMESTAMP WITH TIME ZONE,
    dependencies JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    INDEX idx_assertions_agent (agent_id),
    INDEX idx_assertions_question (question_id),
    INDEX idx_assertions_created (created_at)
);

-- Checkpoints table
CREATE TABLE IF NOT EXISTS checkpoints (
    id VARCHAR(255) PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL REFERENCES agents(id),
    question_id VARCHAR(255) REFERENCES questions(id) ON DELETE CASCADE,
    checkpoint_type backtracking_scope NOT NULL,
    state_data JSONB NOT NULL,
    parent_checkpoint_id VARCHAR(255) REFERENCES checkpoints(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    INDEX idx_checkpoints_agent (agent_id),
    INDEX idx_checkpoints_question (question_id),
    INDEX idx_checkpoints_created (created_at)
);

-- Backtracking events table
CREATE TABLE IF NOT EXISTS backtracking_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question_id VARCHAR(255) NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    scope backtracking_scope NOT NULL,
    trigger_reason TEXT NOT NULL,
    initiating_agent_id VARCHAR(50) REFERENCES agents(id),
    affected_agents TEXT[],
    checkpoint_id VARCHAR(255) REFERENCES checkpoints(id),
    rollback_depth INTEGER DEFAULT 1,
    success BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    INDEX idx_backtracking_question (question_id),
    INDEX idx_backtracking_created (created_at)
);

-- Evidence/retrieval cache table
CREATE TABLE IF NOT EXISTS evidence_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL,
    source VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(query_hash, source),
    INDEX idx_evidence_hash (query_hash),
    INDEX idx_evidence_expires (expires_at)
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    agent_id VARCHAR(50) REFERENCES agents(id),
    question_id VARCHAR(255) REFERENCES questions(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tags JSONB DEFAULT '{}',
    INDEX idx_metrics_type (metric_type),
    INDEX idx_metrics_name (metric_name),
    INDEX idx_metrics_timestamp (timestamp),
    INDEX idx_metrics_agent (agent_id)
);

-- System events/audit log table
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    event_source VARCHAR(255) NOT NULL,
    event_data JSONB NOT NULL,
    severity VARCHAR(20) DEFAULT 'info',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255),
    request_id VARCHAR(255),
    ip_address INET,
    INDEX idx_events_type (event_type),
    INDEX idx_events_created (created_at),
    INDEX idx_events_severity (severity)
);

-- Configuration table
CREATE TABLE IF NOT EXISTS system_configuration (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255)
);

-- Insert default configuration
INSERT INTO system_configuration (key, value, description) VALUES
    ('max_backtrack_depth', '5', 'Maximum depth for backtracking operations'),
    ('question_timeout', '60', 'Default timeout for question processing in seconds'),
    ('cache_ttl', '86400', 'Default cache TTL in seconds'),
    ('token_budget_per_question', '20000', 'Maximum tokens per question')
ON CONFLICT (key) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW v_question_summary AS
SELECT 
    q.id,
    q.question_text,
    q.status,
    q.answer,
    q.confidence,
    q.processing_time,
    q.created_at,
    q.completed_at,
    COUNT(DISTINCT sq.id) as sub_question_count,
    COUNT(DISTINCT be.id) as backtracking_count,
    SUM(CASE WHEN be.scope = 'global' THEN 1 ELSE 0 END) as global_backtrack_count
FROM questions q
LEFT JOIN sub_questions sq ON sq.parent_question_id = q.id
LEFT JOIN backtracking_events be ON be.question_id = q.id
GROUP BY q.id;

CREATE OR REPLACE VIEW v_agent_performance AS
SELECT 
    a.id,
    a.agent_type,
    a.status,
    a.total_tasks_processed,
    a.total_tasks_failed,
    a.reliability_score,
    a.avg_processing_time,
    COUNT(DISTINCT am.id) as messages_sent,
    COUNT(DISTINCT ka.id) as assertions_made,
    COUNT(DISTINCT c.id) as checkpoints_created
FROM agents a
LEFT JOIN agent_messages am ON am.sender_agent_id = a.id AND am.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
LEFT JOIN knowledge_assertions ka ON ka.agent_id = a.id AND ka.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
LEFT JOIN checkpoints c ON c.agent_id = a.id AND c.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY a.id;

-- Create functions
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_questions_updated_at BEFORE UPDATE ON questions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_data(retention_days INTEGER DEFAULT 30)
RETURNS TABLE(
    deleted_questions INTEGER,
    deleted_checkpoints INTEGER,
    deleted_metrics INTEGER,
    deleted_events INTEGER
) AS $$
DECLARE
    cutoff_date TIMESTAMP WITH TIME ZONE;
    q_count INTEGER;
    c_count INTEGER;
    m_count INTEGER;
    e_count INTEGER;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL;
    
    -- Delete old completed questions
    DELETE FROM questions 
    WHERE status IN ('completed', 'failed', 'timeout') 
    AND completed_at < cutoff_date;
    GET DIAGNOSTICS q_count = ROW_COUNT;
    
    -- Delete old checkpoints
    DELETE FROM checkpoints 
    WHERE created_at < cutoff_date;
    GET DIAGNOSTICS c_count = ROW_COUNT;
    
    -- Delete old metrics
    DELETE FROM performance_metrics 
    WHERE timestamp < cutoff_date;
    GET DIAGNOSTICS m_count = ROW_COUNT;
    
    -- Delete old events
    DELETE FROM system_events 
    WHERE created_at < cutoff_date 
    AND severity NOT IN ('error', 'critical');
    GET DIAGNOSTICS e_count = ROW_COUNT;
    
    RETURN QUERY SELECT q_count, c_count, m_count, e_count;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate agent reliability
CREATE OR REPLACE FUNCTION calculate_agent_reliability(agent_id_param VARCHAR(50))
RETURNS FLOAT AS $$
DECLARE
    success_rate FLOAT;
    recent_performance FLOAT;
    backtrack_penalty FLOAT;
    reliability FLOAT;
BEGIN
    -- Calculate basic success rate
    SELECT 
        CASE 
            WHEN total_tasks_processed > 0 
            THEN 1.0 - (total_tasks_failed::FLOAT / total_tasks_processed::FLOAT)
            ELSE 1.0
        END INTO success_rate
    FROM agents
    WHERE id = agent_id_param;
    
    -- Calculate recent performance (last 100 messages)
    SELECT 
        1.0 - (COUNT(CASE WHEN am.message_type = 'reject' THEN 1 END)::FLOAT / 
               GREATEST(COUNT(*)::FLOAT, 1))
    INTO recent_performance
    FROM (
        SELECT message_type 
        FROM agent_messages 
        WHERE sender_agent_id = agent_id_param 
        ORDER BY created_at DESC 
        LIMIT 100
    ) am;
    
    -- Calculate backtracking penalty
    SELECT 
        1.0 - (COUNT(*)::FLOAT / 100.0)
    INTO backtrack_penalty
    FROM backtracking_events
    WHERE initiating_agent_id = agent_id_param
    AND created_at > CURRENT_TIMESTAMP - INTERVAL '7 days';
    
    -- Weighted average
    reliability := (success_rate * 0.4) + (recent_performance * 0.4) + (backtrack_penalty * 0.2);
    
    RETURN GREATEST(0.0, LEAST(1.0, reliability));
END;
$$ LANGUAGE plpgsql;

-- Indexes for performance
CREATE INDEX idx_questions_parent ON questions(parent_question_id) WHERE parent_question_id IS NOT NULL;
CREATE INDEX idx_evidence_cache_cleanup ON evidence_cache(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_metrics_cleanup ON performance_metrics(timestamp);
CREATE INDEX idx_events_cleanup ON system_events(created_at, severity);

-- Permissions (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA reagent TO reagent_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA reagent TO reagent_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA reagent TO reagent_user;

-- Add comments for documentation
COMMENT ON TABLE questions IS 'Main table storing all questions processed by the ReAgent system';
COMMENT ON TABLE agents IS 'Agent registry with performance tracking';
COMMENT ON TABLE checkpoints IS 'Stores system checkpoints for backtracking capability';
COMMENT ON TABLE backtracking_events IS 'Audit trail of all backtracking operations';
COMMENT ON FUNCTION cleanup_old_data IS 'Removes old data based on retention policy';
COMMENT ON FUNCTION calculate_agent_reliability IS 'Calculates reliability score for an agent based on recent performance';
