# config/logging_config.py
import logging
import sys
from pathlib import Path
import json
from datetime import datetime

def setup_logging(log_dir: Path = Path("logs/")):
    """Setup structured logging"""
    log_dir.mkdir(exist_ok=True)
    
    # Structured log formatter
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_object = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            if record.exc_info:
                log_object['exception'] = self.formatException(record.exc_info)
            
            # Add extra fields
            if hasattr(record, 'extra'):
                log_object.update(record.extra)
            
            return json.dumps(log_object)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler for JSON logs
    file_handler = logging.FileHandler(
        log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setFormatter(JSONFormatter())
    
    # Console handler for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Usage in prediction endpoint
def log_prediction(prediction_id, features, result, confidence):
    """Log prediction with structured format"""
    logger = logging.getLogger('prediction')
    extra = {
        'prediction_id': prediction_id,
        'features': features,
        'result': result,
        'confidence': confidence,
        'type': 'prediction'
    }
    logger.info('Prediction made', extra=extra)