import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "NeuroShield",
  description: "Personalized, multi-dimensional stress & recovery dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
