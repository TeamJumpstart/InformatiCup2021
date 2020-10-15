FROM python:3.8

WORKDIR /spe_ed

# Install dependencies
RUN pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy

# Copy code
COPY ["main.py", "./"]

# Entry point
CMD [ "python", "./main.py" ]
