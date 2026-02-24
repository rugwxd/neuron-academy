"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";

export default function GradientDescent3D() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [lr, setLr] = useState(0.1);
  const [steps, setSteps] = useState(0);
  const [path, setPath] = useState<{ x: number; y: number; z: number }[]>([
    { x: 3.5, y: 3.5, z: 0 },
  ]);
  const width = 550;
  const height = 400;

  // Loss function: f(x,y) = x^2 + 3*y^2 + 0.5*x*y - 2*x - 3*y + 5
  const f = (x: number, y: number) =>
    x * x + 3 * y * y + 0.5 * x * y - 2 * x - 3 * y + 5;

  // Gradient
  const grad = (x: number, y: number) => ({
    dx: 2 * x + 0.5 * y - 2,
    dy: 6 * y + 0.5 * x - 3,
  });

  const step = useCallback(() => {
    setPath((prev) => {
      const last = prev[prev.length - 1];
      const g = grad(last.x, last.y);
      const nx = last.x - lr * g.dx;
      const ny = last.y - lr * g.dy;
      return [...prev, { x: nx, y: ny, z: f(nx, ny) }];
    });
    setSteps((s) => s + 1);
  }, [lr]);

  const reset = () => {
    setPath([{ x: 3.5, y: 3.5, z: f(3.5, 3.5) }]);
    setSteps(0);
  };

  const runMultiple = useCallback(() => {
    for (let i = 0; i < 20; i++) {
      setTimeout(() => step(), i * 80);
    }
  }, [step]);

  // Draw the contour plot
  const draw = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([-2, 5]).range([0, w]);
    const yScale = d3.scaleLinear().domain([-1, 4]).range([h, 0]);

    // Generate contour data
    const n = 100;
    const values: number[] = [];
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const x = -2 + (7 * i) / (n - 1);
        const y = -1 + (5 * j) / (n - 1);
        values.push(f(x, y));
      }
    }

    const contours = d3
      .contours()
      .size([n, n])
      .thresholds(d3.range(0, 50, 2))(values);

    const color = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([0, 50]);

    const xContour = d3.scaleLinear().domain([0, n - 1]).range([-2, 5]);
    const yContour = d3.scaleLinear().domain([0, n - 1]).range([-1, 4]);

    const pathGen = d3.geoPath(
      d3.geoTransform({
        point(px: number, py: number) {
          this.stream.point(
            xScale(xContour(px)),
            yScale(yContour(py))
          );
        },
      })
    );

    g.selectAll("path.contour")
      .data(contours)
      .enter()
      .append("path")
      .attr("class", "contour")
      .attr("d", pathGen as unknown as string)
      .attr("fill", (d) => color(d.value))
      .attr("stroke", "#475569")
      .attr("stroke-width", 0.3)
      .attr("opacity", 0.6);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(7))
      .selectAll("text")
      .attr("fill", "#94a3b8")
      .attr("font-size", 10);
    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll("text")
      .attr("fill", "#94a3b8")
      .attr("font-size", 10);

    // Axis labels
    g.append("text").attr("x", w / 2).attr("y", h + 35).attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", 12).text("x");
    g.append("text").attr("x", -35).attr("y", h / 2).attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", 12).attr("transform", `rotate(-90,-35,${h / 2})`).text("y");

    // Draw gradient descent path
    if (path.length > 1) {
      const line = d3
        .line<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y((d) => yScale(d.y));

      g.append("path")
        .datum(path)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", "#f472b6")
        .attr("stroke-width", 2)
        .attr("opacity", 0.8);
    }

    // Draw points
    path.forEach((p, i) => {
      g.append("circle")
        .attr("cx", xScale(p.x))
        .attr("cy", yScale(p.y))
        .attr("r", i === path.length - 1 ? 5 : 2.5)
        .attr("fill", i === path.length - 1 ? "#f472b6" : "#f472b6")
        .attr("opacity", i === path.length - 1 ? 1 : 0.5)
        .attr("stroke", i === path.length - 1 ? "#fff" : "none")
        .attr("stroke-width", 2);
    });

    // Minimum marker
    // Solve: 2x + 0.5y = 2 and 0.5x + 6y = 3
    // x = (2 - 0.5y)/2 = 1 - 0.25y
    // 0.5(1-0.25y) + 6y = 3 => 0.5 - 0.125y + 6y = 3 => 5.875y = 2.5 => y ≈ 0.4255
    // x ≈ 1 - 0.1064 = 0.8936
    g.append("circle")
      .attr("cx", xScale(0.894))
      .attr("cy", yScale(0.426))
      .attr("r", 4)
      .attr("fill", "none")
      .attr("stroke", "#34d399")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "3,2");

    g.selectAll("line,path.domain").attr("stroke", "#475569");
  }, [path]);

  useEffect(() => {
    draw();
  }, [draw]);

  const last = path[path.length - 1];

  return (
    <div>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full max-w-[550px] h-auto rounded-xl border border-border bg-[#0f172a]"
        viewBox={`0 0 ${width} ${height}`}
      />
      <div className="mt-4 flex flex-wrap gap-3 items-center">
        <button
          onClick={step}
          className="px-4 py-2 text-sm rounded-lg bg-accent text-white font-medium hover:bg-accent-hover transition-colors"
        >
          Step
        </button>
        <button
          onClick={runMultiple}
          className="px-4 py-2 text-sm rounded-lg bg-surface border border-border font-medium hover:bg-surface-hover transition-colors"
        >
          Run 20 Steps
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 text-sm rounded-lg bg-surface border border-border font-medium hover:bg-surface-hover transition-colors"
        >
          Reset
        </button>
      </div>
      <div className="mt-3">
        <label className="text-xs font-mono text-muted block mb-1">
          Learning Rate: {lr.toFixed(3)}
        </label>
        <input
          type="range"
          min={0.001}
          max={0.3}
          step={0.005}
          value={lr}
          onChange={(e) => setLr(parseFloat(e.target.value))}
          className="w-48"
        />
      </div>
      <div className="mt-3 text-xs font-mono text-muted space-y-1">
        <div>Step: {steps} | Position: ({last.x.toFixed(3)}, {last.y.toFixed(3)})</div>
        <div>f(x,y) = {f(last.x, last.y).toFixed(4)} | Minimum ≈ f(0.894, 0.426) = {f(0.894, 0.426).toFixed(4)}</div>
      </div>
    </div>
  );
}
