import daisyui from "daisyui";
import typography from "@tailwindcss/typography";

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  prefix: "",
  theme: {
    extend: {},
  },
  safelist: [
    {
      pattern:
        /^(hover:)?text-(light|primary|danger|warning|success|blueberry|mint|neutral)-/,
      variants: ["hover"],
    },
    {
      pattern:
        /^(hover:)?bg-(light|primary|danger|warning|success|blueberry|mint|neutral)-/,
      variants: ["hover"],
    },
    {
      pattern: /^(active:)?bg-light-/,
      variants: ["active"],
    },
    {
      pattern: /^(hover:)?border-light-/,
      variants: ["hover"],
    },
  ],
  plugins: [daisyui, typography],
};
