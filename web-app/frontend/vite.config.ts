import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: true,
    port: 6124,
    allowedHosts: ["mac.sgponte"],
  },
});
