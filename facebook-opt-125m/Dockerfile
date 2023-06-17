FROM python:3.10

# /model/saved - The directory where the model is saved to by this build.
# /model/trained - The directory where further training of the model should save its model to.
# /model/logs - The directory where any relevant logs should be saved to.

RUN mkdir -p /model/saved /model/trained /model/logs
WORKDIR /model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PATH="$PATH:/model/scripts"

# Copy in build dependencies only since the build will take a while.
COPY ./scripts/ ./scripts
COPY src/build.ipynb ./src/
RUN build.sh

COPY ./src/ ./src
COPY ./hack/ ./hack
