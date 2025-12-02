FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci || true
COPY . .
RUN npm run build || true
CMD ["node", "dist/src/index.js"]
