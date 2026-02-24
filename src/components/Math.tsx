"use client";

import katex from "katex";

export function InlineMath({ math }: { math: string }) {
  const html = katex.renderToString(math, {
    throwOnError: false,
    displayMode: false,
  });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export function BlockMath({ math }: { math: string }) {
  const html = katex.renderToString(math, {
    throwOnError: false,
    displayMode: true,
  });
  return (
    <div
      className="my-4 overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
