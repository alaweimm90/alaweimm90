FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
RUN apk add --no-cache curl && \
    chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    touch /var/run/nginx.pid && chown nginx:nginx /var/run/nginx.pid
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
USER nginx
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -fsS http://localhost:8080/health || exit 1
CMD ["nginx", "-g", "daemon off;"]
