# CDN Cache Note

This unrelated note describes CDN cache invalidation.

Cache requirements:

- Use surrogate-key purge for image thumbnail invalidation.
- Keep cache eviction independent from OAuth callback handling.
- Keep cache eviction independent from billing webhook retry behavior.

