"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";

export default function BiasVarianceViz() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [complexity, setComplexity] = useState(3);
  const width = 550;
  const height = 350;

  const draw = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 30, right: 30, bottom: 40, left: 50 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([1, 15]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, 5]).range([h, 0]);

    // Bias² curve: decreases with complexity
    const biasSquared = (c: number) => 4 * Math.exp(-0.3 * c) + 0.1;
    // Variance curve: increases with complexity
    const variance = (c: number) => 0.05 * Math.exp(0.25 * c);
    // Total error
    const totalError = (c: number) => biasSquared(c) + variance(c) + 0.3; // 0.3 = irreducible error

    const xs = d3.range(1, 15.1, 0.2);

    // Irreducible error line
    g.append("line")
      .attr("x1", xScale(1)).attr("y1", yScale(0.3))
      .attr("x2", xScale(15)).attr("y2", yScale(0.3))
      .attr("stroke", "#64748b").attr("stroke-width", 1).attr("stroke-dasharray", "4,4");
    g.append("text").attr("x", w - 5).attr("y", yScale(0.3) - 5)
      .attr("text-anchor", "end").attr("fill", "#64748b").attr("font-size", 10).text("Irreducible Error");

    // Bias² curve
    const biasLine = d3.line<number>().x(c => xScale(c)).y(c => yScale(biasSquared(c))).curve(d3.curveMonotoneX);
    g.append("path").datum(xs).attr("d", biasLine).attr("fill", "none").attr("stroke", "#818cf8").attr("stroke-width", 2.5);

    // Variance curve
    const varLine = d3.line<number>().x(c => xScale(c)).y(c => yScale(variance(c))).curve(d3.curveMonotoneX);
    g.append("path").datum(xs).attr("d", varLine).attr("fill", "none").attr("stroke", "#f472b6").attr("stroke-width", 2.5);

    // Total error curve
    const totalLine = d3.line<number>().x(c => xScale(c)).y(c => yScale(totalError(c))).curve(d3.curveMonotoneX);
    g.append("path").datum(xs).attr("d", totalLine).attr("fill", "none").attr("stroke", "#34d399").attr("stroke-width", 2.5);

    // Find optimal complexity (minimum of total error)
    let optC = 1;
    let optE = totalError(1);
    for (let c = 1; c <= 15; c += 0.1) {
      if (totalError(c) < optE) {
        optE = totalError(c);
        optC = c;
      }
    }

    // Optimal marker
    g.append("circle")
      .attr("cx", xScale(optC)).attr("cy", yScale(optE))
      .attr("r", 5).attr("fill", "#34d399").attr("stroke", "#fff").attr("stroke-width", 2);

    // Current complexity indicator
    g.append("line")
      .attr("x1", xScale(complexity)).attr("y1", 0)
      .attr("x2", xScale(complexity)).attr("y2", h)
      .attr("stroke", "#f59e0b").attr("stroke-width", 2).attr("stroke-dasharray", "6,3");

    // Current values
    const bv = biasSquared(complexity);
    const vv = variance(complexity);
    const tv = totalError(complexity);

    g.append("circle").attr("cx", xScale(complexity)).attr("cy", yScale(bv)).attr("r", 4).attr("fill", "#818cf8").attr("stroke", "#fff").attr("stroke-width", 1.5);
    g.append("circle").attr("cx", xScale(complexity)).attr("cy", yScale(vv)).attr("r", 4).attr("fill", "#f472b6").attr("stroke", "#fff").attr("stroke-width", 1.5);
    g.append("circle").attr("cx", xScale(complexity)).attr("cy", yScale(tv)).attr("r", 4).attr("fill", "#34d399").attr("stroke", "#fff").attr("stroke-width", 1.5);

    // Underfitting / Overfitting labels
    g.append("text").attr("x", xScale(3)).attr("y", 15).attr("text-anchor", "middle").attr("fill", "#818cf8").attr("font-size", 11).attr("font-weight", "bold").text("Underfitting");
    g.append("text").attr("x", xScale(12)).attr("y", 15).attr("text-anchor", "middle").attr("fill", "#f472b6").attr("font-size", 11).attr("font-weight", "bold").text("Overfitting");

    // Axes
    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(7)).selectAll("text").attr("fill", "#94a3b8");
    g.append("g").call(d3.axisLeft(yScale).ticks(5)).selectAll("text").attr("fill", "#94a3b8");
    g.selectAll("line,path.domain").attr("stroke", "#475569");

    g.append("text").attr("x", w / 2).attr("y", h + 35).attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", 12).text("Model Complexity");
    g.append("text").attr("x", -35).attr("y", h / 2).attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", 12).attr("transform", `rotate(-90,-35,${h / 2})`).text("Error");

    // Legend
    const legend = g.append("g").attr("transform", `translate(${w - 130}, 30)`);
    [
      { color: "#818cf8", label: "Bias²" },
      { color: "#f472b6", label: "Variance" },
      { color: "#34d399", label: "Total Error" },
    ].forEach((item, i) => {
      legend.append("line").attr("x1", 0).attr("y1", i * 18).attr("x2", 16).attr("y2", i * 18).attr("stroke", item.color).attr("stroke-width", 2.5);
      legend.append("text").attr("x", 22).attr("y", i * 18 + 4).attr("fill", "#94a3b8").attr("font-size", 11).text(item.label);
    });
  }, [complexity]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full max-w-[550px] h-auto rounded-xl border border-border bg-[#0f172a]"
        viewBox={`0 0 ${width} ${height}`}
      />
      <div className="mt-4">
        <label className="text-sm font-medium block mb-1">
          Model Complexity: {complexity}
        </label>
        <input
          type="range"
          min={1}
          max={15}
          step={0.5}
          value={complexity}
          onChange={(e) => setComplexity(parseFloat(e.target.value))}
          className="w-64"
        />
        <div className="text-xs font-mono text-muted mt-2">
          Bias² = {(4 * Math.exp(-0.3 * complexity) + 0.1).toFixed(3)} |
          Variance = {(0.05 * Math.exp(0.25 * complexity)).toFixed(3)} |
          Total = {(4 * Math.exp(-0.3 * complexity) + 0.1 + 0.05 * Math.exp(0.25 * complexity) + 0.3).toFixed(3)}
        </div>
      </div>
    </div>
  );
}
