import { resolve } from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), wasm()],
  css: {
    postcss: "./postcss.config.js", // Ensure Tailwind is processed here
  },
  server: {
    host: true,
    port: 4173,
    watch: {
      usePolling: true, // Enable polling
      interval: 100,    // Optional: Set polling interval (in milliseconds)
    },
  },
  build: {
    rollupOptions: {
      input: {
        app: resolve(__dirname, 'index.html'),
        appKgChat: resolve(__dirname, 'use-cases/chat-with-kg/index.html'),
        appImport: resolve(__dirname, 'use-cases/unstructured-import/index.html'),
        appReport: resolve(__dirname, 'use-cases/report-generation/index.html'),
      },
    },
  },
})
