import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built files land inside the Python package, which serves them (src/phi_studio/server.py).
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: { outDir: "../src/phi_studio/static", emptyOutDir: true, sourcemap: true },
});
