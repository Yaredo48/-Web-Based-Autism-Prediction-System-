CREATE DATABASE autism_db;

-- Create tables
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE,
    age INTEGER,
    gender VARCHAR(10),
    ethnicity VARCHAR(50),
    jaundice BOOLEAN,
    autism_history BOOLEAN,
    assessment_score JSONB,
    prediction_result VARCHAR(20),
    confidence_score DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    confusion_matrix JSONB,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);