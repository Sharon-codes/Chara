/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export', // Static HTML/JS export capable of running anywhere or deploying to Vercel/GitHub Pages
  images: {
    unoptimized: true,
  },
};

module.exports = nextConfig;
