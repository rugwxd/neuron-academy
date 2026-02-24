import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Neuron Academy — Data Science, ML & AI Mastery",
  description:
    "The most comprehensive free resource for mastering data science, machine learning, and AI. From Python basics to RLHF, explained in plain English with interactive visualizations.",
  keywords: [
    "machine learning",
    "data science",
    "deep learning",
    "AI",
    "transformers",
    "NLP",
    "statistics",
    "Python",
    "interview prep",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
