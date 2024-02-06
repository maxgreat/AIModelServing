#!/bin/bash

# Start RabbitMQ Service
RABBITMQ_STATUS=$(sudo rabbitmqctl status)
if [[ $RABBITMQ_STATUS == *"Error"* ]]; then
  rabbitmq-server start &
  sleep 5 # Wait a bit for RabbitMQ to start
else
  echo "RabbitMQ is running."
fi

# Start Flask App
echo "Starting Flask app..."
export FLASK_APP=main.py
export FLASK_ENV=development
flask run --host 'localhost' --port 8080 --reload & # Runs in background
FLASK_PID=$!

# Start Celery Worker
echo "Starting Celery worker..."
celery -A main.celery worker --loglevel=info --concurrency=1 --pool=solo &
CELERY_PID=$!

# Cleanup function to stop all services when exiting
cleanup() {
    echo "Stopping Flask app..."
    kill $FLASK_PID
    echo "Stopping Celery worker..."
    kill $CELERY_PID
    exit
}

# Trap SIGINT (Ctrl+C) and SIGTERM to cleanup before exiting
trap cleanup SIGINT SIGTERM

# Wait indefinitely until a signal is received
wait