FROM python:3.8.6

WORKDIR /spe_ed

# Install dependencies
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . ./

# Run unit tests
RUN python -m unittest discover -s './tests' -p '*_test.py'

# Entry point
ENTRYPOINT ["python", "./main.py" ]
