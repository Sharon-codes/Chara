/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'brand-bg': '#f8fafc',
        'brand-card': '#ffffff',
        'brand-navy': '#0f172a',
        'brand-blue': '#2563eb',
        'brand-emerald': '#16a34a',
        'brand-crimson': '#dc2626',
        'brand-amber': '#d97706',
        'brand-border': '#e2e8f0',
        'obsidian': '#0B0B0C',
        'laser-cyan': '#00E5FF',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
