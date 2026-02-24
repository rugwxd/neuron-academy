import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/neuron-academy",
  pageExtensions: ["js", "jsx", "md", "mdx", "ts", "tsx"],
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
