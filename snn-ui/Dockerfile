FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install && find /app/node_modules -name '*.sh' -type f -exec sed -i 's/\r$//' {} +

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
