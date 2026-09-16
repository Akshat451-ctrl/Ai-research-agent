import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Proxy API calls to the FastAPI backend during development, so the
    // frontend can call fetch("/research") with no CORS setup needed on
    // either side - both look like the same origin to the browser.
    proxy: {
      '/research': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
