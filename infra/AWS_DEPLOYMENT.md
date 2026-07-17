# AWS Deployment Guide

Target architecture:

```
Route53 (DNS) -> CloudFront (CDN/TLS) -> ALB -> ECS Fargate service (FastAPI container)
                                                     |-> Amazon OpenSearch Service (vector + BM25)
                                                     |-> Amazon S3 (raw files, page images, frames)
                                                     |-> MongoDB (Atlas, or self-managed on EC2/DocumentDB)
```

## 1. Container registry
```bash
aws ecr create-repository --repository-name multimodal-rag
docker build -t multimodal-rag .
docker tag multimodal-rag:latest <account_id>.dkr.ecr.<region>.amazonaws.com/multimodal-rag:latest
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
docker push <account_id>.dkr.ecr.<region>.amazonaws.com/multimodal-rag:latest
```

## 2. S3
- Create a bucket for raw uploads + derived assets (page renders, video frames).
- Set `s3_bucket` in `app/config.py` / env var.
- Grant the ECS task role `s3:PutObject`, `s3:GetObject` on that bucket.

## 3. Amazon OpenSearch Service (Vector Engine)
- Create a managed OpenSearch domain (or OpenSearch Serverless with a vector
  collection) with the k-NN plugin enabled — it's on by default in the
  managed service.
- Put the domain in the same VPC as the ECS tasks (or use a VPC endpoint) so
  traffic doesn't cross the public internet.
- Update `opensearch_host`, `opensearch_port` (443 for managed HTTPS
  endpoint), `opensearch_use_ssl=true`, and IAM/basic-auth credentials.

## 4. MongoDB (long-term memory)
- Simplest: MongoDB Atlas with a VPC peering connection into your AWS VPC.
- Alternative: Amazon DocumentDB (Mongo-compatible) if you want to stay
  fully inside AWS IAM/VPC boundaries — same `pymongo` client code works
  with a compatible connection string.

## 5. ECS Fargate + ALB
- Task definition: point at the ECR image, expose container port 8000,
  attach an IAM task role with S3 + OpenSearch + Secrets Manager access.
- Store secrets (Mongo URI, OpenSearch creds, Anthropic API key) in AWS
  Secrets Manager and inject as environment variables in the task def.
- Create an Application Load Balancer with a target group pointing at the
  ECS service on port 8000. Health check path: `/health`.
- For multi-task deployments, note the code comment in
  `app/memory/short_term.py` — `InMemorySaver` is per-process, so either
  enable ALB sticky sessions per `thread_id`, or swap in a shared
  checkpointer backend (Redis/Postgres) before scaling beyond one task.

## 6. CloudFront + Route53
- CloudFront distribution with the ALB as its origin — gives you TLS
  termination at the edge, caching for any static/read-mostly endpoints,
  and DDoS protection via AWS Shield.
- Route53: create an alias record for your domain pointing at the
  CloudFront distribution.

## 7. CI/CD (optional, recommended)
- GitHub Actions (or CodePipeline): on push to `main`, build & push the
  Docker image to ECR, then run `aws ecs update-service --force-new-deployment`.

## 8. Evaluation in the loop
- Run `app/evaluation/ragas_eval.py` against a held-out Q&A set as part of
  CI, or schedule it (e.g. EventBridge + a Fargate task) to periodically
  score live traffic samples and push RAGAS metrics to CloudWatch for
  regression alerting.
