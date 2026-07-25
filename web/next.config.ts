import type { NextConfig } from "next";

// Published as a GitHub Pages project site, which serves from a subpath
// (/fpl-engine) and cannot run a Node server — so this is a fully static export
// and all data is read from a baked JSON file rather than an API.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
