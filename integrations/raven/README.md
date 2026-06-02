# Raven Integration

This directory contains adapters for using RavenStorm runtime objects through
`agent_core`.

The adapter intentionally keeps imports from `app.openai_agents_runtime` and
`app.tools` out of the core package. Run these tests only from a RavenStorm
checkout or an environment that exposes Raven runtime modules.
