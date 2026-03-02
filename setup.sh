#!/bin/bash

# Create folders
mkdir -p app/api app/core app/models app/services app/database app/utils
mkdir -p data logs tests

# Create __init__ files
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py
touch app/models/__init__.py
touch app/services/__init__.py
touch app/database/__init__.py
touch app/utils/__init__.py

# Create main files
touch app/main.py
touch app/config.py
touch app/api/routes.py
touch app/api/schemas.py
touch app/core/logger.py
touch app/core/exceptions.py
touch app/core/error_handlers.py
touch app/models/train.py
touch app/models/predict.py
touch app/services/feature_engineering.py
touch app/database/connection.py
touch app/database/queries.py
touch app/utils/helpers.py

# Root files
touch requirements.txt Dockerfile README.md .env .gitignore

# Install dependencies
pip install -r requirements.txt

echo "Project structure created successfully."