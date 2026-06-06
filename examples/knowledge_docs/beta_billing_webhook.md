# Beta Billing Webhook Retry

The beta billing webhook delivery path uses a tenant-scoped retry budget.

Operational requirements:

- Allow a 30 request per minute burst budget per tenant.
- Treat HTTP 429 responses as retryable pressure signals.
- Use jittered backoff for retry scheduling.
- Include idempotency keys so duplicate delivery can be detected safely.

