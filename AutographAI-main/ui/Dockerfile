FROM node:lts-alpine

ARG FOLDER=/app

# Create and set working directory
RUN mkdir -p $FOLDER
WORKDIR $FOLDER

# Install dependencies first (use package.json to cache dependencies)
COPY package.json package-lock.json ./
RUN npm install

# Copy the application source code
COPY . ./

# Expose the Vite dev server port
EXPOSE 4173

# Use development mode by default
CMD ["npm", "run", "dev"]
