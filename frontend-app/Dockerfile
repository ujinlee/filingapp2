# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for lxml, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional Python dependencies
RUN pip install inflect

# Copy the rest of the code
COPY . .

# Expose port for Cloud Run
ENV PORT=8080

# Debug: show directory and test import
RUN ls -l /app
RUN ls -l /app/multi_tool_agent
RUN cat /app/multi_tool_agent/__init__.py
RUN python -c "import multi_tool_agent.agent"

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "multi_tool_agent.main:app", "--host", "0.0.0.0", "--port", "8080"] 