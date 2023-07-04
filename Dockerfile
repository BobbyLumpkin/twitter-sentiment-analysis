# Pull python3.9 container.
FROM python:3.9

# Set environment vars, working dir and expose port.
ENV PYTHONUNBUFFERED 1 
EXPOSE 8501
WORKDIR /app

# Copy project and install dependencies from requirements.txt.
COPY . .
RUN pip install -r requirements.txt
RUN pip install .

# Run the application on port 8501.
CMD cd app;streamlit run Home.py --server.port 8501
