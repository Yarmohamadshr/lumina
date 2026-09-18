# The agent service. Built from the REPO ROOT as context, because it depends on the shared
# @lumina/contract workspace:  fly deploy --config backend/agent/fly.toml --dockerfile backend/agent/Dockerfile .
FROM node:24-alpine AS build
WORKDIR /app

# Manifests first, so a code change does not re-run npm ci from scratch (Docker layer cache).
COPY package.json package-lock.json tsconfig.base.json ./
COPY packages/contract/package.json packages/contract/
COPY backend/agent/package.json backend/agent/
COPY backend/gateway/package.json backend/gateway/
COPY web/package.json web/
RUN npm ci --ignore-scripts

COPY packages/contract packages/contract
COPY backend/agent backend/agent
RUN npm run build -w @lumina/contract && npm run build -w @lumina/agent

# --- runtime: only what is needed to run
FROM node:24-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/node_modules node_modules
COPY --from=build /app/package.json package.json
COPY --from=build /app/packages/contract/package.json packages/contract/package.json
COPY --from=build /app/packages/contract/dist packages/contract/dist
COPY --from=build /app/backend/agent/package.json backend/agent/package.json
COPY --from=build /app/backend/agent/dist backend/agent/dist

# env.ts resolves paths from the working directory, so run from the service folder, as in dev.
WORKDIR /app/backend/agent
EXPOSE 8000
CMD ["node", "dist/index.js"]
