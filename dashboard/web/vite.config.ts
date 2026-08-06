import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      // Nitro server from dashboard/server, see README for how to run it.
      "/api": { target: "http://localhost:3099", changeOrigin: true },
    },
  },
})
