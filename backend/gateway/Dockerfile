# The gateway. Built from the REPO ROOT as context (shared @lumina/contract workspace):
#   fly deploy --config backend/gateway/fly.toml --dockerfile backend/gateway/Dockerfile .
FROM node:24-alpine AS build
WORKDIR /app

COPY package.json package-lock.json tsconfig.base.json ./
COPY packages/contract/package.json packages/contract/
COPY backend/agent/package.json backend/agent/
COPY backend/gateway/package.json backend/gateway/
COPY web/package.json web/
RUN npm ci --ignore-scripts

COPY packages/contract packages/contract
COPY backend/gateway backend/gateway
RUN npm run build -w @lumina/contract && npm run build -w @lumina/gateway

# --- runtime
FROM node:24-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/node_modules node_modules
COPY --from=build /app/package.json package.json
COPY --from=build /app/packages/contract/package.json packages/contract/package.json
COPY --from=build /app/packages/contract/dist packages/contract/dist
COPY --from=build /app/backend/gateway/package.json backend/gateway/package.json
COPY --from=build /app/backend/gateway/dist backend/gateway/dist

WORKDIR /app/backend/gateway
EXPOSE 8787
CMD ["node", "dist/index.js"]
