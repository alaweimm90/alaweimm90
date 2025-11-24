## Objectives

- Conduct an internet research survey of Managed Cloud Platform Services (MCPS) and compile a structured, versioned document covering providers, services, pricing, SLAs, integration, benchmarks, security/compliance, and a decision framework.

## Scope

- Providers: AWS, Microsoft Azure, Google Cloud, IBM Cloud, Oracle Cloud, Alibaba Cloud, Tencent Cloud, Huawei Cloud, DigitalOcean, Linode/Akamai, Vultr, OVHcloud, SAP BTP, Cloudflare, Fastly, Heroku, Netlify, Vercel, Fly.io.
- Service categories: compute, storage, networking/CDN/DNS, databases, data/analytics, AI/ML, messaging/queues, security/IAM/KMS, observability, DevOps/CI/CD, serverless/edge.

## Research Method

- Primary sources: official docs, pricing pages, SLA pages, compliance portals, service catalogs, architectural guides.
- Secondary sources: analyst reports (Gartner/CNCF), third-party benchmark summaries; note benchmarks are workload-specific.
- For each provider, collect:
  - Service catalog summary by category
  - Pricing model highlights (pay-as-you-go, reserved/committed, spot/preemptible, free tiers)
  - SLAs (availability %, region/service notes)
  - Integration requirements (auth/IAM, SDKs/CLIs, network setup/VPC, endpoints)
  - Security/compliance certifications (ISO 27001, SOC 1/2/3, PCI DSS, HIPAA eligible, FedRAMP/GDPR)
  - References/links (URLs) for verification.

## Document Structure

- MCPS Overview
- Provider Profiles (one section per provider)
- Comparison Matrix
  - Columns: Provider, Category, Key Services, Features, Pricing model, SLA, Benchmarks refs, Certifications, Integration notes
- Decision Framework
  - Inputs: technical reqs, cost, scalability, geo, compliance
  - Scoring rubric and example scenarios
- Implementation Considerations
- Maintenance & Monitoring Guidelines
- Appendices: Source links, update workflow
- Versioning: maintain in repository under `docs/cloud/mcps.md` with changelog and date/version header.

## Comparison Criteria & Rubric

- Technical fit (feature coverage, managed vs serverless options)
- Cost-effectiveness (TCO estimate, discounts, free tiers)
- Scalability (auto-scaling, global footprint, multi-region)
- Geo availability (regions/locations; data residency)
- Compliance/security (required certifications)
- Integration complexity (VPC/IAM/SDK), support maturity.

## Output & Artifacts

- Structured document (Markdown) with matrix tables and decision framework.
- Source-of-truth JSON/YAML catalog of services to facilitate updates.
- Update cadence: quarterly refresh; CI reminder issue.

## Next Steps (after approval)

1. Perform provider-by-provider research and extract verified data, linking sources.
2. Draft provider profiles and fill the matrix.
3. Write decision framework and integration considerations.
4. Add maintenance/monitoring guidelines per service category.
5. Commit document with version header and changelog entry; add CI reminder.

## Risks & Notes

- Benchmark variability: include references and caution; do not generalize performance claims.
- Rapid pricing changes: link pricing calculators; note date of capture.
- Compliance scopes vary by service/region: list explicitly when known.

Approve to begin research and compile the document; I will produce the structured matrix and decision framework with linked sources and versioned outputs.
