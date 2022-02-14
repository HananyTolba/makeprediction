FROM python:3.7-slim
COPY . /package
RUN cd /package
RUN python -m pip install --upgrade pip
WORKDIR /package
RUN pip install .
# CMD ['python', 'run.py']

