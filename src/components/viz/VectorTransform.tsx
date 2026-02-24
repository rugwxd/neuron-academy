"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";

export default function VectorTransform() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [matrix, setMatrix] = useState({ a: 1, b: 0, c: 0, d: 1 });
  const width = 500;
  const height = 500;
  const scale = 40;

  const draw = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const cx = width / 2;
    const cy = height / 2;

    // Grid
    const g = svg.append("g");
    for (let i = -6; i <= 6; i++) {
      g.append("line")
        .attr("x1", cx + i * scale)
        .attr("y1", 0)
        .attr("x2", cx + i * scale)
        .attr("y2", height)
        .attr("stroke", "#334155")
        .attr("stroke-width", 0.5);
      g.append("line")
        .attr("x1", 0)
        .attr("y1", cy + i * scale)
        .attr("x2", width)
        .attr("y2", cy + i * scale)
        .attr("stroke", "#334155")
        .attr("stroke-width", 0.5);
    }

    // Axes
    g.append("line").attr("x1", 0).attr("y1", cy).attr("x2", width).attr("y2", cy).attr("stroke", "#64748b").attr("stroke-width", 1);
    g.append("line").attr("x1", cx).attr("y1", 0).attr("x2", cx).attr("y2", height).attr("stroke", "#64748b").attr("stroke-width", 1);

    // Transformed grid lines (subtle)
    const { a, b, c, d } = matrix;
    for (let i = -6; i <= 6; i++) {
      // Vertical grid line through (i, t) for t in [-6,6]
      const x1t = a * i + b * (-6);
      const y1t = c * i + d * (-6);
      const x2t = a * i + b * 6;
      const y2t = c * i + d * 6;
      g.append("line")
        .attr("x1", cx + x1t * scale)
        .attr("y1", cy - y1t * scale)
        .attr("x2", cx + x2t * scale)
        .attr("y2", cy - y2t * scale)
        .attr("stroke", "#818cf8")
        .attr("stroke-width", 0.5)
        .attr("opacity", 0.3);
      // Horizontal
      const hx1 = a * (-6) + b * i;
      const hy1 = c * (-6) + d * i;
      const hx2 = a * 6 + b * i;
      const hy2 = c * 6 + d * i;
      g.append("line")
        .attr("x1", cx + hx1 * scale)
        .attr("y1", cy - hy1 * scale)
        .attr("x2", cx + hx2 * scale)
        .attr("y2", cy - hy2 * scale)
        .attr("stroke", "#818cf8")
        .attr("stroke-width", 0.5)
        .attr("opacity", 0.3);
    }

    // Original basis vectors
    const drawArrow = (x: number, y: number, color: string, label: string, dashed = false) => {
      g.append("line")
        .attr("x1", cx)
        .attr("y1", cy)
        .attr("x2", cx + x * scale)
        .attr("y2", cy - y * scale)
        .attr("stroke", color)
        .attr("stroke-width", dashed ? 1.5 : 2.5)
        .attr("stroke-dasharray", dashed ? "6,3" : "none")
        .attr("marker-end", "none");
      // Arrowhead
      const len = Math.sqrt(x * x + y * y);
      if (len > 0.01) {
        const ux = x / len;
        const uy = y / len;
        const tipX = cx + x * scale;
        const tipY = cy - y * scale;
        const arrLen = 10;
        const arrAngle = Math.PI / 6;
        const cos = Math.cos(arrAngle);
        const sin = Math.sin(arrAngle);
        g.append("polygon")
          .attr("points", [
            [tipX, tipY],
            [tipX - arrLen * (ux * cos + uy * sin), tipY + arrLen * (-ux * sin + uy * cos)],
            [tipX - arrLen * (ux * cos - uy * sin), tipY + arrLen * (ux * sin + uy * cos)],
          ].map(p => p.join(",")).join(" "))
          .attr("fill", color)
          .attr("opacity", dashed ? 0.5 : 1);
      }
      g.append("text")
        .attr("x", cx + x * scale + 10)
        .attr("y", cy - y * scale - 5)
        .attr("fill", color)
        .attr("font-size", 12)
        .attr("font-weight", "bold")
        .text(label);
    };

    // Original e1, e2 (dashed)
    drawArrow(1, 0, "#94a3b8", "e₁", true);
    drawArrow(0, 1, "#94a3b8", "e₂", true);

    // Transformed basis
    drawArrow(a, c, "#f472b6", `[${a.toFixed(1)}, ${c.toFixed(1)}]`);
    drawArrow(b, d, "#34d399", `[${b.toFixed(1)}, ${d.toFixed(1)}]`);

    // Determinant area (parallelogram)
    const det = a * d - b * c;
    g.append("polygon")
      .attr("points", [
        [cx, cy],
        [cx + a * scale, cy - c * scale],
        [cx + (a + b) * scale, cy - (c + d) * scale],
        [cx + b * scale, cy - d * scale],
      ].map(p => p.join(",")).join(" "))
      .attr("fill", det >= 0 ? "#818cf8" : "#f472b6")
      .attr("opacity", 0.15);

    // Determinant label
    g.append("text")
      .attr("x", 10)
      .attr("y", height - 10)
      .attr("fill", "#94a3b8")
      .attr("font-size", 13)
      .attr("font-family", "monospace")
      .text(`det = ${det.toFixed(2)}`);
  }, [matrix]);

  useEffect(() => {
    draw();
  }, [draw]);

  const presets = [
    { label: "Identity", m: { a: 1, b: 0, c: 0, d: 1 } },
    { label: "Rotation 45°", m: { a: 0.71, b: -0.71, c: 0.71, d: 0.71 } },
    { label: "Scale 2x", m: { a: 2, b: 0, c: 0, d: 2 } },
    { label: "Shear", m: { a: 1, b: 1, c: 0, d: 1 } },
    { label: "Reflection", m: { a: -1, b: 0, c: 0, d: 1 } },
    { label: "Projection", m: { a: 1, b: 0, c: 0, d: 0 } },
  ];

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-4">
        {presets.map((p) => (
          <button
            key={p.label}
            onClick={() => setMatrix(p.m)}
            className="px-3 py-1 text-xs rounded-lg bg-surface border border-border hover:bg-surface-hover transition-colors font-medium"
          >
            {p.label}
          </button>
        ))}
      </div>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full max-w-[500px] h-auto rounded-xl border border-border bg-[#0f172a]"
        viewBox={`0 0 ${width} ${height}`}
      />
      <div className="mt-4 grid grid-cols-2 gap-3 max-w-xs">
        {(["a", "b", "c", "d"] as const).map((key) => (
          <div key={key}>
            <label className="text-xs font-mono text-muted block mb-1">
              {key === "a" ? "a (row1,col1)" : key === "b" ? "b (row1,col2)" : key === "c" ? "c (row2,col1)" : "d (row2,col2)"}
            </label>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.1}
              value={matrix[key]}
              onChange={(e) =>
                setMatrix((prev) => ({ ...prev, [key]: parseFloat(e.target.value) }))
              }
              className="w-full"
            />
            <span className="text-xs font-mono">{matrix[key].toFixed(1)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
