FROM python:3.8

WORKDIR /spe_ed

# Install dependencies
RUN pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy --ignore-pipfile

# Copy code
COPY . ./

# Run unit tests
RUN pipenv run python -m unittest discover -s './tests' -p '*_test.py'

# Entry point
CMD ["pipenv", "run", "python", "./main.py" ]
