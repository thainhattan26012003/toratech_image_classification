FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install --production

COPY server.js .
COPY public ./public

EXPOSE 3003

CMD ["node", "server.js"]
