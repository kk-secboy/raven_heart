# Alpha Callback-State Nonce

The alpha OAuth callback flow uses callback-state nonce `alpha-42`.

Security requirements:

- Generate the nonce before redirecting to the authorization endpoint.
- Persist the nonce with an expiry before the redirect leaves the service.
- Reject replay of the same nonce before exchanging the authorization code for tokens.
- Treat token exchange as blocked until callback-state replay validation succeeds.

