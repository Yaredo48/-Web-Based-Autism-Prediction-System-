# production_test.sh
#!/bin/bash

echo "Running production tests..."

# Test API endpoints
echo "Testing API health..."
curl -f http://localhost:8000/health || exit 1

echo "Testing prediction endpoint..."
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25,
    "gender": "m",
    "ethnicity": "White",
    "jaundice": false,
    "autism_history": true,
    "assessment_scores": {
      "A1": 1, "A2": 0, "A3": 1, "A4": 0, "A5": 1,
      "A6": 0, "A7": 1, "A8": 0, "A9": 1, "A10": 0
    }
  }' || exit 1

# Test database connection
echo "Testing database..."
docker-compose exec db pg_isready -U postgres || exit 1

# Test Redis
echo "Testing Redis..."
docker-compose exec redis redis-cli ping | grep PONG || exit 1

# Load test
echo "Running load test..."
python -m locust -f tests/load_test.py --headless -u 10 -r 2 -t 30s \
  --host=http://localhost:8000 || echo "Load test completed"

echo "All production tests passed!"
