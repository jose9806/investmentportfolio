FROM public.ecr.aws/lambda/python:3.9

# Install dependencies using file requirements.txt
COPY requirements.txt .
RUN  pip install -r requirements.txt

# Copy function code
COPY . ${LAMBDA_TASK_ROOT}/

# Set the CMD to handler
CMD ["MPT_Script_v2.handler"]