FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements-gpu-cu124.lock /app/
RUN python -m pip install --no-cache-dir -r requirements-gpu-cu124.lock
COPY flytaiko /app/flytaiko
COPY training /app/training
COPY taiko /app/taiko
COPY flyconnectome /app/flyconnectome
COPY models/v25/config.json /app/models/v25/config.json
COPY web-fly/public /app/web-fly/public
RUN mkdir -p /app/runs /app/connectome_data /app/web-fly/public/demos
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "flytaiko.replay_web_server", "--bind", "0.0.0.0", "--port", "8000"]
