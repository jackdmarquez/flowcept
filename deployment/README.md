# Flowcept Local Services

## MongoDB Credentials

MongoDB Compose deployments require credentials through an ignored env file.

```bash
cp deployment/mongo.env.example deployment/mongo.env
$EDITOR deployment/mongo.env
```

Change `MONGO_INITDB_ROOT_PASSWORD` before starting MongoDB. The same file includes
`MONGO_URI`, which Flowcept can use when it runs on the host:

```bash
set -a
. deployment/mongo.env
set +a

docker compose -f deployment/compose-mongo.yml up -d
```

For Kubernetes, create an equivalent Secret with `MONGO_INITDB_ROOT_USERNAME`,
`MONGO_INITDB_ROOT_PASSWORD`, and `MONGO_URI`, then mount or expose those values
to the MongoDB and Flowcept pods.

## Apple Silicon

The Compose files set:

```yaml
platform: ${FLOWCEPT_DOCKER_PLATFORM:-linux/amd64}
```

This default works on Mac M1/M2 through Docker Desktop emulation and avoids
manifest issues with images that do not publish `linux/arm64` variants.

For Mongo/Redis-only testing, you can use native Apple Silicon images:

```bash
export FLOWCEPT_DOCKER_PLATFORM=linux/arm64
docker compose -f deployment/compose-mongo.yml up -d
```

Keep the default `linux/amd64` for the Kafka and Grafana Compose files unless
you have verified all referenced images support `linux/arm64`.
