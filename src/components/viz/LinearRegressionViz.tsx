"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";

interface Point {
  x: number;
  y: number;
}

export default function LinearRegressionViz() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [points, setPoints] = useState<Point[]>([
    { x: 1, y: 2.1 },
    { x: 2, y: 3.8 },
    { x: 3, y: 5.2 },
    { x: 4, y: 7.1 },
    { x: 5, y: 8.5 },
    { x: 6, y: 10.8 },
    { x: 7, y: 12.3 },
    { x: 1.5, y: 3.5 },
    { x: 3.5, y: 6.5 },
    { x: 5.5, y: 9.0 },
  ]);
  const [showResiduals, setShowResiduals] = useState(true);
  const width = 550;
  const height = 400;

  // Compute OLS
  const computeRegression = useCallback((pts: Point[]) => {
    const n = pts.length;
    if (n < 2) return { slope: 0, intercept: 0, r2: 0 };
    const sumX = pts.reduce((s, p) => s + p.x, 0);
    const sumY = pts.reduce((s, p) => s + p.y, 0);
    const sumXY = pts.reduce((s, p) => s + p.x * p.y, 0);
    const sumX2 = pts.reduce((s, p) => s + p.x * p.x, 0);
    const meanY = sumY / n;
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    const ssTot = pts.reduce((s, p) => s + (p.y - meanY) ** 2, 0);
    const ssRes = pts.reduce((s, p) => s + (p.y - (slope * p.x + intercept)) ** 2, 0);
    const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
    return { slope, intercept, r2 };
  }, []);

  const draw = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xDomain = [0, 9];
    const yDomain = [0, 16];
    const xScale = d3.scaleLinear().domain(xDomain).range([0, w]);
    const yScale = d3.scaleLinear().domain(yDomain).range([h, 0]);

    // Grid
    for (let i = 0; i <= 9; i++) {
      g.append("line").attr("x1", xScale(i)).attr("y1", 0).attr("x2", xScale(i)).attr("y2", h).attr("stroke", "#1e293b").attr("stroke-width", 0.5);
    }
    for (let i = 0; i <= 16; i += 2) {
      g.append("line").attr("x1", 0).attr("y1", yScale(i)).attr("x2", w).attr("y2", yScale(i)).attr("stroke", "#1e293b").attr("stroke-width", 0.5);
    }

    // Axes
    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(9)).selectAll("text").attr("fill", "#94a3b8");
    g.append("g").call(d3.axisLeft(yScale).ticks(8)).selectAll("text").attr("fill", "#94a3b8");
    g.selectAll("line,path.domain").attr("stroke", "#475569");

    const { slope, intercept } = computeRegression(points);

    // Regression line
    const x1 = 0;
    const x2 = 9;
    g.append("line")
      .attr("x1", xScale(x1))
      .attr("y1", yScale(slope * x1 + intercept))
      .attr("x2", xScale(x2))
      .attr("y2", yScale(slope * x2 + intercept))
      .attr("stroke", "#818cf8")
      .attr("stroke-width", 2.5);

    // Residuals
    if (showResiduals) {
      points.forEach((p) => {
        const predicted = slope * p.x + intercept;
        g.append("line")
          .attr("x1", xScale(p.x))
          .attr("y1", yScale(p.y))
          .attr("x2", xScale(p.x))
          .attr("y2", yScale(predicted))
          .attr("stroke", "#f472b6")
          .attr("stroke-width", 1.5)
          .attr("stroke-dasharray", "4,2")
          .attr("opacity", 0.7);
      });
    }

    // Points (draggable)
    points.forEach((p, i) => {
      g.append("circle")
        .attr("cx", xScale(p.x))
        .attr("cy", yScale(p.y))
        .attr("r", 6)
        .attr("fill", "#34d399")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .attr("cursor", "grab")
        .call(
          d3.drag<SVGCircleElement, unknown>()
            .on("drag", (event) => {
              const nx = xScale.invert(event.x);
              const ny = yScale.invert(event.y);
              setPoints((prev) => {
                const next = [...prev];
                next[i] = {
                  x: Math.max(0.1, Math.min(8.9, nx)),
                  y: Math.max(0.1, Math.min(15.9, ny)),
                };
                return next;
              });
            })
        );
    });
  }, [points, showResiduals, computeRegression]);

  useEffect(() => {
    draw();
  }, [draw]);

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX - 50;
    const my = (e.clientY - rect.top) * scaleY - 20;
    const xScale = d3.scaleLinear().domain([0, 9]).range([0, width - 70]);
    const yScale = d3.scaleLinear().domain([0, 16]).range([height - 60, 0]);
    const x = xScale.invert(mx);
    const y = yScale.invert(my);
    if (x >= 0.1 && x <= 8.9 && y >= 0.1 && y <= 15.9) {
      setPoints((prev) => [...prev, { x, y }]);
    }
  };

  const { slope, intercept, r2 } = computeRegression(points);

  return (
    <div>
      <p className="text-sm text-muted mb-3">
        Click to add points. Drag points to move them. Watch the regression line update in real-time.
      </p>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full max-w-[550px] h-auto rounded-xl border border-border bg-[#0f172a] cursor-crosshair"
        viewBox={`0 0 ${width} ${height}`}
        onClick={handleSvgClick}
      />
      <div className="mt-4 flex flex-wrap gap-4 items-center">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={showResiduals}
            onChange={(e) => setShowResiduals(e.target.checked)}
            className="rounded"
          />
          Show residuals
        </label>
        <button
          onClick={() =>
            setPoints([
              { x: 1, y: 2.1 },
              { x: 2, y: 3.8 },
              { x: 3, y: 5.2 },
              { x: 4, y: 7.1 },
              { x: 5, y: 8.5 },
              { x: 6, y: 10.8 },
              { x: 7, y: 12.3 },
              { x: 1.5, y: 3.5 },
              { x: 3.5, y: 6.5 },
              { x: 5.5, y: 9.0 },
            ])
          }
          className="px-3 py-1 text-xs rounded-lg bg-surface border border-border hover:bg-surface-hover transition-colors"
        >
          Reset Points
        </button>
        <button
          onClick={() => setPoints([])}
          className="px-3 py-1 text-xs rounded-lg bg-surface border border-border hover:bg-surface-hover transition-colors"
        >
          Clear All
        </button>
      </div>
      <div className="mt-3 text-xs font-mono text-muted">
        <div>y = {slope.toFixed(3)}x + {intercept.toFixed(3)}</div>
        <div>R² = {r2.toFixed(4)} | n = {points.length}</div>
      </div>
    </div>
  );
}
