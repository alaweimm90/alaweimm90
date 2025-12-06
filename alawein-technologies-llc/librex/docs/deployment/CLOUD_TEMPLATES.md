# Cloud Deployment Templates for Librex.QAP-new

Comprehensive templates and scripts for deploying Librex.QAP-new to major cloud platforms.

---

## AWS EC2 Deployment

### 1. IAM Role Setup

```bash
# Create IAM role for EC2 instance
aws iam create-role --role-name Librex.QAP-EC2-Role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach policies
aws iam attach-role-policy --role-name Librex.QAP-EC2-Role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy

aws iam attach-role-policy --role-name Librex.QAP-EC2-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
```

### 2. Launch Template

```bash
# Create security group
aws ec2 create-security-group \
  --group-name Librex.QAP-sg \
  --description "Security group for Librex.QAP"

# Allow HTTP/HTTPS/SSH
aws ec2 authorize-security-group-ingress \
  --group-name Librex.QAP-sg \
  --protocol tcp --port 22 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-name Librex.QAP-sg \
  --protocol tcp --port 80 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-name Librex.QAP-sg \
  --protocol tcp --port 443 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-name Librex.QAP-sg \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0
```

### 3. User Data Script

Save as `aws-user-data.sh`:

```bash
#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
apt-get install -y docker.io docker-compose git

# Start Docker
systemctl start docker
systemctl enable docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Clone repository
cd /home/ubuntu
git clone https://github.com/AlaweinOS/AlaweinOS.git
cd AlaweinOS/Librex.QAP-new

# Build and run with Docker Compose
docker-compose up -d

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb

# Setup CloudWatch monitoring
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<EOF
{
  "metrics": {
    "namespace": "Librex.QAP",
    "metrics_collected": {
      "cpu": {
        "measurement": [{"name": "cpu_usage_idle"}],
        "totalcpu": false
      },
      "mem": {
        "measurement": [{"name": "mem_used_percent"}]
      },
      "disk": {
        "measurement": [{"name": "used_percent"}],
        "metrics_collection_interval": 60,
        "resources": ["/"]
      }
    }
  }
}
EOF

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a query -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s

echo "Librex.QAP deployment complete on $(date)" > /home/ubuntu/deployment.log
```

### 4. Launch Instance

```bash
#!/bin/bash

INSTANCE_TYPE="t3.medium"  # 2 vCPU, 4GB RAM
AMI_ID="ami-0c55b159cbfafe1f0"  # Ubuntu 22.04 LTS
KEY_NAME="my-Librex.QAP-key"
SECURITY_GROUP="Librex.QAP-sg"
REGION="us-east-1"

# Create key pair if needed
aws ec2 create-key-pair \
  --key-name $KEY_NAME \
  --region $REGION \
  --query 'KeyMaterial' \
  --output text > ${KEY_NAME}.pem

chmod 400 ${KEY_NAME}.pem

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-groups $SECURITY_GROUP \
  --user-data file://aws-user-data.sh \
  --region $REGION \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Launched instance: $INSTANCE_ID"

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --region $REGION \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance is running at: $PUBLIC_IP"
echo "SSH: ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "API: http://${PUBLIC_IP}:8000"
```

---

## Google Cloud Run Deployment

### 1. Setup

```bash
PROJECT_ID="my-Librex.QAP-project"
REGION="us-central1"
SERVICE_NAME="Librex.QAP-api"

gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 2. Prepare Dockerfile for Cloud Run

Save as `Dockerfile.cloudrun`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

RUN apt-get update && apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir gunicorn

EXPOSE $PORT

CMD exec gunicorn --bind :$PORT --workers 4 --threads 2 --worker-class gthread app:app
```

### 3. Build and Deploy

```bash
#!/bin/bash

PROJECT_ID="my-Librex.QAP-project"
REGION="us-central1"
SERVICE_NAME="Librex.QAP-api"
IMAGE_NAME="Librex.QAP-api"

# Build with Cloud Build
gcloud builds submit . \
  --tag gcr.io/${PROJECT_ID}/${IMAGE_NAME} \
  --timeout=1200s

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/${PROJECT_ID}/${IMAGE_NAME} \
  --platform managed \
  --region $REGION \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --allow-unauthenticated

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --format 'value(status.url)')

echo "Deployed to: $SERVICE_URL"
echo "Monitor: gcloud run services describe $SERVICE_NAME --region $REGION"
```

### 4. Setup Cloud Scheduler for Benchmarks

```bash
#!/bin/bash

SCHEDULER_JOB="Librex.QAP-benchmark"
REGION="us-central1"
SERVICE_URL="https://your-cloud-run-url"

# Create scheduled job (weekly benchmarks)
gcloud scheduler jobs create http $SCHEDULER_JOB \
  --location=$REGION \
  --schedule="0 0 * * 0" \
  --http-method=POST \
  --uri="${SERVICE_URL}/benchmark" \
  --message-body='{"type":"comprehensive","instances":["nug12","nug20","tai30a"]}' \
  --oidc-service-account-email=your-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --oidc-token-audience=$SERVICE_URL
```

---

## Azure Container Instances

### 1. Resource Group and Registry

```bash
RESOURCE_GROUP="Librex.QAP-rg"
REGISTRY_NAME="Librex.QAPregistry"
LOCATION="eastus"

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Create container registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $REGISTRY_NAME \
  --sku Basic
```

### 2. Build and Push Image

```bash
#!/bin/bash

RESOURCE_GROUP="Librex.QAP-rg"
REGISTRY_NAME="Librex.QAPregistry"
IMAGE_NAME="Librex.QAP-new"
IMAGE_TAG="latest"

# Build image
az acr build \
  --registry $REGISTRY_NAME \
  --image ${IMAGE_NAME}:${IMAGE_TAG} \
  .

# List images
az acr repository list --name $REGISTRY_NAME

# Get login credentials
az acr credential show \
  --name $REGISTRY_NAME \
  --query "[username,passwords[0].value]" \
  --output tsv
```

### 3. Deploy Container Instance

```bash
#!/bin/bash

RESOURCE_GROUP="Librex.QAP-rg"
REGISTRY_NAME="Librex.QAPregistry"
CONTAINER_NAME="Librex.QAP-api"
IMAGE_NAME="Librex.QAP-new"
IMAGE_TAG="latest"

# Get registry login server
REGISTRY_LOGIN_SERVER=$(az acr show \
  --name $REGISTRY_NAME \
  --query loginServer \
  --output tsv)

# Create container instance
az container create \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --image ${REGISTRY_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG} \
  --registry-login-server $REGISTRY_LOGIN_SERVER \
  --registry-username $(az acr credential show --name $REGISTRY_NAME --query username --output tsv) \
  --registry-password $(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" --output tsv) \
  --port 8000 \
  --cpu 2 \
  --memory 2 \
  --environment-variables PYTHONUNBUFFERED=1

# Get container properties
az container show \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --query "{FQDN:ipAddress.fqdn, ProvisioningState:provisioningState}"
```

### 4. Setup Monitoring

```bash
#!/bin/bash

RESOURCE_GROUP="Librex.QAP-rg"
CONTAINER_NAME="Librex.QAP-api"
WORKSPACE_NAME="Librex.QAP-workspace"
LOCATION="eastus"

# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $WORKSPACE_NAME \
  --location $LOCATION

# Get workspace ID
WORKSPACE_ID=$(az monitor log-analytics workspace show \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $WORKSPACE_NAME \
  --query id \
  --output tsv)

# Configure diagnostics
az monitor diagnostic-settings create \
  --name Librex.QAP-diagnostics \
  --resource $WORKSPACE_ID \
  --logs '[{"category":"ContainerInstanceLogs","enabled":true}]' \
  --metrics '[{"category":"AllMetrics","enabled":true}]'
```

---

## Kubernetes Deployment (Optional)

### 1. Kubernetes Manifest

Save as `k8s-deployment.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: Librex.QAP

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: Librex.QAP-api
  namespace: Librex.QAP
spec:
  replicas: 3
  selector:
    matchLabels:
      app: Librex.QAP-api
  template:
    metadata:
      labels:
        app: Librex.QAP-api
    spec:
      containers:
      - name: Librex.QAP
        image: gcr.io/my-project/Librex.QAP-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: PYTHONUNBUFFERED
          value: "1"

---
apiVersion: v1
kind: Service
metadata:
  name: Librex.QAP-service
  namespace: Librex.QAP
spec:
  selector:
    app: Librex.QAP-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: Librex.QAP-hpa
  namespace: Librex.QAP
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: Librex.QAP-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 2. Deploy to Kubernetes

```bash
# GKE
gcloud container clusters create Librex.QAP-cluster --zone us-central1-a
kubectl apply -f k8s-deployment.yaml
kubectl port-forward svc/Librex.QAP-service 8000:80 -n Librex.QAP

# EKS
eksctl create cluster --name Librex.QAP-cluster --region us-east-1
kubectl apply -f k8s-deployment.yaml

# AKS
az aks create --resource-group Librex.QAP-rg --name Librex.QAP-cluster
kubectl apply -f k8s-deployment.yaml
```

---

## Monitoring & Logging Setup

### 1. CloudWatch Dashboard (AWS)

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create custom dashboard
dashboard_body = {
    'widgets': [
        {
            'type': 'metric',
            'properties': {
                'metrics': [
                    ['Librex.QAP', 'APILatency', {'stat': 'Average'}],
                    ['.', 'APIErrors', {'stat': 'Sum'}],
                    ['.', 'OptimizationTime', {'stat': 'Average'}],
                    ['AWS/EC2', 'CPUUtilization'],
                    ['.', 'NetworkIn'],
                ]
            }
        }
    ]
}

cloudwatch.put_dashboard(
    DashboardName='Librex.QAP-Dashboard',
    DashboardBody=str(dashboard_body)
)
```

### 2. Application Performance Monitoring

Save as `app_monitoring.py`:

```python
import logging
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
api_requests = Counter(
    'Librex.QAP_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_latency = Histogram(
    'Librex.QAP_api_latency_seconds',
    'API latency in seconds',
    ['endpoint']
)

optimization_time = Histogram(
    'Librex.QAP_optimization_seconds',
    'Optimization time in seconds',
    ['method']
)

active_optimizations = Gauge(
    'Librex.QAP_active_optimizations',
    'Number of active optimizations'
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def monitor_api_call(endpoint):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                api_requests.labels(
                    method='POST',
                    endpoint=endpoint,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                api_requests.labels(
                    method='POST',
                    endpoint=endpoint,
                    status='error'
                ).inc()
                logger.error(f"Error in {endpoint}: {str(e)}")
                raise
            finally:
                duration = time.time() - start
                api_latency.labels(endpoint=endpoint).observe(duration)
        return wrapper
    return decorator
```

---

## Deployment Checklist

- [ ] Configure authentication (API keys, database credentials)
- [ ] Set up monitoring and alerting
- [ ] Configure auto-scaling policies
- [ ] Test failover and recovery
- [ ] Document access procedures
- [ ] Set up log aggregation
- [ ] Configure backups
- [ ] Test disaster recovery
- [ ] Monitor costs and optimize
- [ ] Document deployment architecture

---

**Last Updated:** November 2024
