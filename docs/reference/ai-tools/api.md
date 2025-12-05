# REST API Reference

> HTTP REST API for AI Tools

## Base URL

```
http://localhost:3200
```

## Endpoints

| Method | Path              | Description           |
| ------ | ----------------- | --------------------- |
| GET    | /health           | Health check          |
| GET    | /compliance/score | Get compliance report |
| POST   | /compliance/check | Run compliance check  |
| GET    | /security/report  | Get security report   |
| POST   | /security/scan    | Run security scan     |
| GET    | /cache/stats      | Get cache statistics  |
| DELETE | /cache            | Clear cache           |
| GET    | /monitor/status   | Get monitor status    |
| GET    | /errors           | List errors           |
| GET    | /issues           | List issues           |
| GET    | /metrics          | Get AI metrics        |
| POST   | /sync             | Sync context          |
| GET    | /dashboard        | Get ASCII dashboard   |

## Examples

### Get Compliance Score

```bash
curl http://localhost:3200/compliance/score
```

### Run Security Scan

```bash
curl -X POST http://localhost:3200/security/scan
```

### List Critical Issues

```bash
curl http://localhost:3200/issues/critical
```
